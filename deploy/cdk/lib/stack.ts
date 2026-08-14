import * as cdk from "aws-cdk-lib";
import { Construct } from "constructs";
import * as dynamodb from "aws-cdk-lib/aws-dynamodb";
import * as bedrock from "aws-cdk-lib/aws-bedrock";
import * as iam from "aws-cdk-lib/aws-iam";
import * as ec2 from "aws-cdk-lib/aws-ec2";
import * as ecs from "aws-cdk-lib/aws-ecs";
import * as ecs_patterns from "aws-cdk-lib/aws-ecs-patterns";
import * as servicediscovery from "aws-cdk-lib/aws-servicediscovery";
import * as logs from "aws-cdk-lib/aws-logs";

/**
 * Infrastructure for the privacy-preserving document AI sample:
 *  - DynamoDB table for PII token -> original mappings (short TTL)
 *  - DynamoDB table for the optional sLLM result cache
 *  - A Bedrock Guardrail (+ a published version) that detects PII
 *  - A least-privilege managed policy the services attach to their task role
 *  - An ECS Fargate deployment of the three CPU services (ocr-service,
 *    pii-service, orchestrator), reachable at the orchestrator's public ALB
 *  - (opt-in, `-c enableGpuOcr=true`) a GPU-backed ECS EC2 service running
 *    the paddleocr-vl OCR engine, off by default because it costs real money
 *    per hour it runs regardless of traffic
 *
 * Container images are built from source (`ecs.ContainerImage.fromAsset`) —
 * `cdk deploy` builds and pushes them to a CDK-managed ECR repo itself, so
 * there is no pre-existing image reference to keep in sync or lose access to.
 *
 * NOT in scope: the vLLM sLLM endpoint (provide your own GPU host — this is
 * unrelated to the GPU OCR path above, which is a different model).
 */
export class PrivacyPreservingDocAiStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    const piiTable = new dynamodb.Table(this, "PiiMappings", {
      tableName: "pii-mappings",
      partitionKey: { name: "session_id", type: dynamodb.AttributeType.STRING },
      sortKey: { name: "token", type: dynamodb.AttributeType.STRING },
      timeToLiveAttribute: "ttl",
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      removalPolicy: cdk.RemovalPolicy.DESTROY, // sample only — drop on stack delete
    });

    const qwenCache = new dynamodb.Table(this, "SllmResultCache", {
      tableName: "qwen-cache",
      partitionKey: { name: "cache_key", type: dynamodb.AttributeType.STRING },
      timeToLiveAttribute: "expires_at",
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });

    const guardrail = new bedrock.CfnGuardrail(this, "PiiGuardrail", {
      name: "privacy-preserving-doc-ai",
      blockedInputMessaging: "Input blocked by guardrail.",
      blockedOutputsMessaging: "Output blocked by guardrail.",
      sensitiveInformationPolicyConfig: {
        piiEntitiesConfig: [
          { type: "NAME", action: "ANONYMIZE" },
          { type: "EMAIL", action: "ANONYMIZE" },
          { type: "PHONE", action: "ANONYMIZE" },
          { type: "ADDRESS", action: "ANONYMIZE" },
          { type: "CREDIT_DEBIT_CARD_NUMBER", action: "ANONYMIZE" },
          { type: "US_SOCIAL_SECURITY_NUMBER", action: "ANONYMIZE" },
        ],
        regexesConfig: [
          {
            name: "KR_RRN",
            pattern: "\\d{6}-\\d{7}",
            action: "ANONYMIZE",
            description: "Korean resident registration number",
          },
          {
            name: "KR_ACCOUNT",
            // Korean bank account numbers are hyphen-segmented, and the
            // segment lengths vary by bank (3, 4, or up to 16+ digits in the
            // first segment; 1-4 trailing segments). This alternation covers
            // the common shapes without also matching a plain calendar date
            // (which CREDIT_DEBIT_CARD_NUMBER-style single "\d+-\d+" patterns
            // tend to do).
            pattern:
              "(\\d{7,16}-\\d{2,6}-\\d{2,14}(-\\d{1,14})*|\\d{5,6}-\\d{2,6}-\\d{2,14}(-\\d{1,14})*|\\d{3,4}-\\d{3,6}-\\d{2,14}(-\\d{1,14})*|\\d{4}-\\d{2}-\\d{3,14}(-\\d{1,14})*|\\d{3}-\\d{2}-\\d{4,14}(-\\d{1,14})*)",
            action: "ANONYMIZE",
            description: "Korean bank account number (hyphen-segmented, several bank formats)",
          },
        ],
      },
    });

    // A numbered version, not DRAFT — DRAFT can change without a deploy, which
    // would silently change what Stage 4 verifies against. pii-service also
    // refuses to run against an unversioned Guardrail (see its GUARDRAIL_ID /
    // GUARDRAIL_VERSION fail-closed check) for the same reason.
    const guardrailVersion = new bedrock.CfnGuardrailVersion(this, "PiiGuardrailVersion", {
      guardrailIdentifier: guardrail.attrGuardrailId,
    });

    const appPolicy = new iam.ManagedPolicy(this, "AppPolicy", {
      statements: [
        new iam.PolicyStatement({
          actions: [
            "dynamodb:PutItem",
            "dynamodb:BatchWriteItem",
            "dynamodb:Query",
            "dynamodb:GetItem",
          ],
          resources: [piiTable.tableArn, qwenCache.tableArn],
        }),
        new iam.PolicyStatement({
          // Narrow this to your specific model / inference-profile ARN in production.
          actions: ["bedrock:InvokeModel"],
          resources: ["*"],
        }),
        new iam.PolicyStatement({
          actions: ["bedrock:ApplyGuardrail"],
          resources: [guardrail.attrGuardrailArn],
        }),
      ],
    });

    new cdk.CfnOutput(this, "PiiTableName", { value: piiTable.tableName });
    new cdk.CfnOutput(this, "QwenCacheTableName", { value: qwenCache.tableName });
    new cdk.CfnOutput(this, "GuardrailId", { value: guardrail.attrGuardrailId });
    new cdk.CfnOutput(this, "AppPolicyArn", { value: appPolicy.managedPolicyArn });

    // ── ECS Fargate: run the three CPU services and expose the orchestrator ──
    //
    // No NAT Gateway (its ~$32/month flat fee would dwarf everything else in
    // this sample) — tasks sit in public subnets with a public IP for their
    // own internet egress (pulling images, reaching Bedrock/DynamoDB/ECR).
    // That is an *outbound* path, not inbound exposure: each task's security
    // group allows no inbound traffic except from the ALB (orchestrator) or
    // from the orchestrator's own security group (ocr-service, pii-service).
    // Internal service-to-service calls resolve via ECS Service Connect
    // (Cloud Map DNS scoped to the cluster's namespace), not the public ALB.
    const vpc = new ec2.Vpc(this, "Vpc", {
      maxAzs: 2,
      natGateways: 0,
      subnetConfiguration: [{ name: "public", subnetType: ec2.SubnetType.PUBLIC, cidrMask: 24 }],
    });

    const namespace = new servicediscovery.PrivateDnsNamespace(this, "ServiceConnectNamespace", {
      name: "internal",
      vpc,
    });

    const cluster = new ecs.Cluster(this, "Cluster", { vpc });

    const taskRole = new iam.Role(this, "TaskRole", {
      assumedBy: new iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
      managedPolicies: [appPolicy],
    });

    const logGroup = new logs.LogGroup(this, "ServiceLogs", {
      retention: logs.RetentionDays.ONE_WEEK,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });

    // Internal-only Fargate service (no public IP needed on the *listener*
    // side — reached over Service Connect — but does need one for its own
    // egress, since there's no NAT). Returns the service so the caller can
    // wire allowFrom() rules.
    const addInternalService = (
      name: string,
      dockerfilePath: string,
      containerPort: number,
      environment: Record<string, string>,
    ): ecs.FargateService => {
      const taskDef = new ecs.FargateTaskDefinition(this, `${name}TaskDef`, {
        cpu: 512,
        memoryLimitMiB: 1024,
        taskRole,
      });
      taskDef.addContainer(`${name}Container`, {
        image: ecs.ContainerImage.fromAsset(dockerfilePath),
        environment,
        portMappings: [{ name: "http", containerPort, appProtocol: ecs.AppProtocol.http }],
        logging: ecs.LogDrivers.awsLogs({ logGroup, streamPrefix: name }),
      });
      return new ecs.FargateService(this, `${name}Service`, {
        cluster,
        taskDefinition: taskDef,
        desiredCount: 1,
        // desiredCount is 1, so a 50%-healthy rolling deployment can't be
        // maintained during updates anyway — 0 avoids CDK's warning about it
        // without changing actual behavior for a single-task service.
        minHealthyPercent: 0,
        circuitBreaker: { rollback: true },
        assignPublicIp: true, // egress only — inbound is SG-restricted below
        vpcSubnets: { subnetType: ec2.SubnetType.PUBLIC },
        serviceConnectConfiguration: {
          namespace: namespace.namespaceName,
          services: [{ portMappingName: "http", dnsName: name, port: containerPort }],
        },
      });
    };

    const ocrService = addInternalService("ocr-service", "../../src/ocr-service", 8083, {
      OCR_ENGINE: "paddleocr",
      OCR_LANG: "korean",
    });

    const piiService = addInternalService("pii-service", "../../src/pii-service", 8082, {
      AWS_REGION: this.region,
      PII_TABLE: piiTable.tableName,
      GUARDRAIL_ID: guardrail.attrGuardrailId,
      GUARDRAIL_VERSION: guardrailVersion.attrVersion,
    });

    const orchestratorTaskDef = new ecs.FargateTaskDefinition(this, "OrchestratorTaskDef", {
      cpu: 512,
      memoryLimitMiB: 1024,
      taskRole,
    });
    orchestratorTaskDef.addContainer("OrchestratorContainer", {
      image: ecs.ContainerImage.fromAsset("../../src/orchestrator"),
      environment: {
        AWS_REGION: this.region,
        MODEL_ID: "global.anthropic.claude-sonnet-5",
        OCR_SERVICE_URL: `http://${ocrService.serviceName}:8083`,
        PII_SERVICE_URL: `http://${piiService.serviceName}:8082`,
        QWEN_CACHE_TABLE: qwenCache.tableName,
        GUARDRAIL_ID: guardrail.attrGuardrailId,
        GUARDRAIL_VERSION: guardrailVersion.attrVersion,
        // Point this at YOUR vLLM endpoint — not created by this stack (see
        // the README's "Prerequisites"). Left as the compose-file default
        // (unreachable from inside AWS) so Stage 3 fails loudly rather than
        // silently pointing nowhere in particular.
        SLLM_ENDPOINT: "http://localhost:8000",
      },
      portMappings: [{ containerPort: 8080 }],
      logging: ecs.LogDrivers.awsLogs({ logGroup, streamPrefix: "orchestrator" }),
    });

    const orchestrator = new ecs_patterns.ApplicationLoadBalancedFargateService(this, "OrchestratorAlb", {
      cluster,
      taskDefinition: orchestratorTaskDef,
      desiredCount: 1,
      minHealthyPercent: 0, // see the internal services' comment above
      circuitBreaker: { rollback: true },
      publicLoadBalancer: true,
      assignPublicIp: true,
      taskSubnets: { subnetType: ec2.SubnetType.PUBLIC },
    });
    // The L3 pattern above doesn't expose serviceConnectConfiguration directly —
    // enable it on the underlying FargateService construct instead. The
    // orchestrator doesn't need a dnsName of its own here (nothing calls it
    // over Service Connect; it's reached via the public ALB), just membership
    // in the namespace so it can resolve ocr-service/pii-service by name.
    orchestrator.service.enableServiceConnect({ namespace: namespace.namespaceName });
    // Same reasoning as the class-level comment: only the ALB's SG may reach
    // the orchestrator task; only the orchestrator's SG may reach the two
    // internal services (Service Connect's own SG rules handle the ALB<->task
    // hop, this covers task<->task).
    ocrService.connections.allowFrom(orchestrator.service, ec2.Port.tcp(8083));
    piiService.connections.allowFrom(orchestrator.service, ec2.Port.tcp(8082));

    new cdk.CfnOutput(this, "OrchestratorUrl", { value: `http://${orchestrator.loadBalancer.loadBalancerDnsName}` });

    // ── GPU OCR (opt-in only — `cdk deploy -c enableGpuOcr=true`) ──
    //
    // paddleocr-vl needs a GPU; Fargate doesn't support GPUs, so this is an
    // EC2-backed capacity provider instead. It bills by the hour the instance
    // is up, not by request — leaving this off by default is the difference
    // between $0 and a real per-hour charge for a sample nobody asked to run.
    if (this.node.tryGetContext("enableGpuOcr") === "true" || this.node.tryGetContext("enableGpuOcr") === true) {
      cluster.addCapacity("GpuCapacity", {
        instanceType: new ec2.InstanceType("g4dn.xlarge"),
        machineImage: ecs.EcsOptimizedImage.amazonLinux2(ecs.AmiHardwareType.GPU),
        minCapacity: 1,
        maxCapacity: 1,
        vpcSubnets: { subnetType: ec2.SubnetType.PUBLIC },
      });

      // AWS_VPC network mode (not the EC2 launch type's bridge-mode default)
      // so this gets its own ENI + security group, the same model Fargate
      // tasks use above — otherwise the container port maps to a random host
      // port and a fixed-port security group rule below would be wrong.
      const gpuTaskDef = new ecs.Ec2TaskDefinition(this, "PaddleVlTaskDef", {
        networkMode: ecs.NetworkMode.AWS_VPC,
      });
      gpuTaskDef.addContainer("PaddleVlContainer", {
        image: ecs.ContainerImage.fromAsset("../ocr-gpu"),
        cpu: 2048,
        memoryLimitMiB: 8192,
        gpuCount: 1,
        portMappings: [{ name: "http", containerPort: 8080, appProtocol: ecs.AppProtocol.http }],
        logging: ecs.LogDrivers.awsLogs({ logGroup, streamPrefix: "paddleocr-vl" }),
      });
      const paddleVlService = new ecs.Ec2Service(this, "PaddleVlService", {
        cluster,
        taskDefinition: gpuTaskDef,
        desiredCount: 1,
        vpcSubnets: { subnetType: ec2.SubnetType.PUBLIC },
        serviceConnectConfiguration: {
          namespace: namespace.namespaceName,
          services: [{ portMappingName: "http", dnsName: "paddleocr-vl", port: 8080 }],
        },
      });
      paddleVlService.connections.allowFrom(ocrService, ec2.Port.tcp(8080));
      // ocr-service picks this up by setting OCR_ENGINE=paddleocr-vl and
      // PADDLE_VL_ENDPOINT=http://paddleocr-vl:8080 on its task definition —
      // left as a manual follow-up edit rather than auto-wired, so turning
      // this on doesn't silently change ocr-service's default engine.
    }
  }
}
