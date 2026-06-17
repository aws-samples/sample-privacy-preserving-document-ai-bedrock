#!/usr/bin/env node
import * as cdk from "aws-cdk-lib";
import { PrivacyPreservingDocAiStack } from "../lib/stack";

const app = new cdk.App();
new PrivacyPreservingDocAiStack(app, "PrivacyPreservingDocAiStack", {
  env: {
    account: process.env.CDK_DEFAULT_ACCOUNT,
    region: process.env.CDK_DEFAULT_REGION,
  },
});
