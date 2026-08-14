import * as cdk from "aws-cdk-lib";
import { Construct } from "constructs";
import * as dynamodb from "aws-cdk-lib/aws-dynamodb";
import * as bedrock from "aws-cdk-lib/aws-bedrock";
import * as iam from "aws-cdk-lib/aws-iam";

/**
 * Minimal infrastructure for the privacy-preserving document AI sample:
 *  - DynamoDB table for PII token -> original mappings (short TTL)
 *  - DynamoDB table for the optional sLLM result cache
 *  - A Bedrock Guardrail that detects PII (built-in entities + a KR RRN regex)
 *  - A least-privilege managed policy the services would attach to their role
 *
 * NOT in scope: the vLLM sLLM endpoint (provide your own GPU host), networking,
 * and the container runtime (run via docker-compose or your own ECS/EKS setup).
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
  }
}
