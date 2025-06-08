"""
Deployment Script for Gold Price Prediction System

This script handles deployment of the gold price prediction system
to various environments (local, staging, production).
"""

import os
import sys
import subprocess
import logging
import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import docker
import requests
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.utils import setup_logging
from src.mlflow_integration import MLflowManager
from config.config import load_config


logger = logging.getLogger(__name__)


class DeploymentManager:
    """
    Manages deployment operations for the gold price prediction system
    """
    
    def __init__(self, environment: str = "development"):
        """
        Initialize deployment manager
        
        Args:
            environment: Target environment (development, staging, production)
        """
        self.environment = environment
        self.config = load_config()
        self.docker_client = docker.from_env()
        self.project_root = project_root
        
        # Environment-specific configurations
        self.env_configs = {
            "development": {
                "api_port": 8000,
                "mlflow_port": 5000,
                "replicas": 1,
                "resources": {
                    "cpu_limit": "0.5",
                    "memory_limit": "1g"
                }
            },
            "staging": {
                "api_port": 8000,
                "mlflow_port": 5000,
                "replicas": 2,
                "resources": {
                    "cpu_limit": "1",
                    "memory_limit": "2g"
                }
            },
            "production": {
                "api_port": 8000,
                "mlflow_port": 5000,
                "replicas": 3,
                "resources": {
                    "cpu_limit": "2",
                    "memory_limit": "4g"
                }
            }
        }
        
        logger.info(f"Initialized deployment manager for {environment}")
    
    def check_prerequisites(self) -> bool:
        """
        Check if all prerequisites are met for deployment
        
        Returns:
            True if all prerequisites are met
        """
        logger.info("Checking deployment prerequisites...")
        
        checks = []
        
        # Check Docker
        try:
            self.docker_client.ping()
            checks.append(("Docker", True, "Docker is running"))
        except Exception as e:
            checks.append(("Docker", False, f"Docker error: {e}"))
        
        # Check required files
        required_files = [
            "Dockerfile",
            "docker-compose.yml",
            "requirements.txt",
            "config/config.py"
        ]
        
        for file_path in required_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                checks.append((f"File {file_path}", True, "File exists"))
            else:
                checks.append((f"File {file_path}", False, "File missing"))
        
        # Check model files
        model_dir = self.project_root / "models" / "saved_models"
        if model_dir.exists() and list(model_dir.glob("*.h5")):
            checks.append(("Trained model", True, "Model files found"))
        else:
            checks.append(("Trained model", False, "No trained models found"))
        
        # Print results
        all_passed = True
        for check_name, passed, message in checks:
            status = "✓" if passed else "✗"
            logger.info(f"{status} {check_name}: {message}")
            if not passed:
                all_passed = False
        
        return all_passed
    
    def build_image(self, tag: Optional[str] = None) -> str:
        """
        Build Docker image for the application
        
        Args:
            tag: Tag for the Docker image
            
        Returns:
            Image tag
        """
        if tag is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            tag = f"gold-prediction:{self.environment}_{timestamp}"
        
        logger.info(f"Building Docker image: {tag}")
        
        try:
            # Build image
            image, build_logs = self.docker_client.images.build(
                path=str(self.project_root),
                tag=tag,
                rm=True,
                forcerm=True
            )
            
            # Log build output
            for log in build_logs:
                if 'stream' in log:
                    logger.debug(log['stream'].strip())
            
            logger.info(f"Successfully built image: {tag}")
            return tag
            
        except Exception as e:
            logger.error(f"Failed to build image: {e}")
            raise
    
    def deploy_local(self, image_tag: str) -> Dict[str, Any]:
        """
        Deploy to local environment using Docker Compose
        
        Args:
            image_tag: Docker image tag
            
        Returns:
            Deployment status
        """
        logger.info("Deploying to local environment...")
        
        try:
            # Set environment variables
            env_vars = {
                "GOLD_API_IMAGE": image_tag,
                "ENVIRONMENT": self.environment,
                "API_PORT": str(self.env_configs[self.environment]["api_port"]),
                "MLFLOW_PORT": str(self.env_configs[self.environment]["mlflow_port"])
            }
            
            # Update environment
            os.environ.update(env_vars)
            
            # Run docker-compose
            compose_file = self.project_root / "docker-compose.yml"
            cmd = [
                "docker-compose",
                "-f", str(compose_file),
                "up", "-d",
                "--build"
            ]
            
            result = subprocess.run(
                cmd,
                cwd=str(self.project_root),
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.info("Local deployment successful")
                return {
                    "status": "success",
                    "environment": self.environment,
                    "services": self._get_running_services(),                "endpoints": self._get_service_endpoints()
                }
            else:
                logger.error(f"Local deployment failed: {result.stderr}")
                raise Exception(f"Docker Compose failed: {result.stderr}")
                
        except Exception as e:
            logger.error(f"Local deployment error: {e}")
            raise
    
    def _get_running_services(self) -> List[Dict[str, Any]]:
        """Get list of running Docker services"""
        try:
            containers = self.docker_client.containers.list()
            services = []
            for container in containers:
                services.append({
                    "name": container.name,
                    "image": container.image.tags[0] if container.image.tags else "unknown",
                    "status": container.status,
                    "ports": container.attrs.get("NetworkSettings", {}).get("Ports", {})
                })
            return services
        except Exception as e:
            logger.warning(f"Failed to get running services: {e}")
            return []
    
    def _get_service_endpoints(self) -> Dict[str, str]:
        """Get service endpoints for local deployment"""
        try:
            api_port = self.env_configs[self.environment]["api_port"]
            mlflow_port = self.env_configs[self.environment]["mlflow_port"]
            
            return {
                "api": f"http://localhost:{api_port}",
                "mlflow": f"http://localhost:{mlflow_port}",
                "health": f"http://localhost:{api_port}/health",
                "docs": f"http://localhost:{api_port}/docs"
            }
        except Exception as e:
            logger.warning(f"Failed to get service endpoints: {e}")
            return {}
    
    def deploy_cloud(self, provider: str, image_tag: Optional[str] = None) -> Dict[str, Any]:
        """
        Deploy to cloud provider
        
        Args:
            provider: Cloud provider (aws, gcp, azure)
            image_tag: Docker image tag
            
        Returns:
            Deployment status
        """
        logger.info(f"Deploying to {provider.upper()} cloud...")
          if provider.lower() == "aws":
            if image_tag is None:
                image_tag = self.build_image()
            return self._deploy_aws(image_tag)
        elif provider.lower() == "gcp":
            if image_tag is None:
                image_tag = self.build_image()
            return self._deploy_gcp(image_tag)
        elif provider.lower() == "azure":
            if image_tag is None:
                image_tag = self.build_image()
            return self._deploy_azure(image_tag)
        else:
            raise ValueError(f"Unsupported cloud provider: {provider}")
    
    def _deploy_aws(self, image_tag: str) -> Dict[str, Any]:
        """Deploy to AWS using ECS/Fargate and RDS"""
        logger.info("Deploying to AWS...")
        
        try:
            import boto3
            from botocore.exceptions import ClientError
        except ImportError:
            raise ImportError("boto3 required for AWS deployment. Install with: pip install boto3")
        
        try:
            # Initialize AWS clients
            ecs_client = boto3.client('ecs')
            ecr_client = boto3.client('ecr')
            rds_client = boto3.client('rds')
            
            # AWS configuration
            aws_config = {
                "cluster_name": f"gold-prediction-{self.environment}",
                "service_name": f"gold-api-{self.environment}",
                "task_definition": f"gold-prediction-task-{self.environment}",
                "ecr_repository": "gold-price-prediction",
                "db_instance_id": f"gold-db-{self.environment}",
                "region": os.getenv("AWS_REGION", "us-east-1")
            }
            
            # 1. Push image to ECR
            ecr_uri = self._push_to_ecr(ecr_client, aws_config["ecr_repository"], image_tag)
            
            # 2. Create/Update RDS instance
            db_endpoint = self._setup_aws_database(rds_client, aws_config)
            
            # 3. Create/Update ECS task definition
            task_def_arn = self._create_ecs_task_definition(
                ecs_client, aws_config, ecr_uri, db_endpoint
            )
            
            # 4. Create/Update ECS service
            service_arn = self._create_ecs_service(
                ecs_client, aws_config, task_def_arn
            )
            
            # 5. Wait for deployment to complete
            self._wait_for_ecs_deployment(ecs_client, aws_config)
            
            # 6. Get service endpoint
            endpoint = self._get_aws_service_endpoint(aws_config)
            
            return {
                "status": "success",
                "provider": "aws",
                "cluster": aws_config["cluster_name"],
                "service": service_arn,
                "endpoint": endpoint,
                "image": ecr_uri,
                "database": db_endpoint
            }
            
        except Exception as e:
            logger.error(f"AWS deployment failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    def _push_to_ecr(self, ecr_client, repository_name: str, image_tag: str) -> str:
        """Push Docker image to ECR"""
        logger.info("Pushing image to ECR...")
        
        try:
            # Get ECR login token
            response = ecr_client.get_authorization_token()
            token = response['authorizationData'][0]['authorizationToken']
            endpoint = response['authorizationData'][0]['proxyEndpoint']
            
            # Login to ECR
            import base64
            username, password = base64.b64decode(token).decode().split(':')
            
            # Tag and push image
            ecr_uri = f"{endpoint.replace('https://', '')}/{repository_name}:{image_tag}"
            
            # Tag local image
            subprocess.run([
                "docker", "tag", image_tag, ecr_uri
            ], check=True)
            
            # Login to ECR
            subprocess.run([
                "docker", "login", "--username", username, "--password", password, endpoint
            ], check=True)
            
            # Push image
            subprocess.run([
                "docker", "push", ecr_uri
            ], check=True)
            
            logger.info(f"Image pushed to ECR: {ecr_uri}")
            return ecr_uri
            
        except Exception as e:
            logger.error(f"Failed to push to ECR: {e}")
            raise
    
    def _setup_aws_database(self, rds_client, aws_config: Dict) -> str:
        """Setup RDS PostgreSQL instance"""
        logger.info("Setting up RDS database...")
        
        try:
            # Check if database exists
            try:
                response = rds_client.describe_db_instances(
                    DBInstanceIdentifier=aws_config["db_instance_id"]
                )
                db_instance = response['DBInstances'][0]
                
                if db_instance['DBInstanceStatus'] == 'available':
                    logger.info("Database already exists and is available")
                    return db_instance['Endpoint']['Address']
                    
            except rds_client.exceptions.DBInstanceNotFoundFault:
                # Create new database
                logger.info("Creating new RDS instance...")
                
                rds_client.create_db_instance(
                    DBInstanceIdentifier=aws_config["db_instance_id"],
                    DBInstanceClass='db.t3.micro',
                    Engine='postgres',
                    MasterUsername='golduser',
                    MasterUserPassword=os.getenv('DB_PASSWORD', 'SecurePass123!'),
                    AllocatedStorage=20,
                    VpcSecurityGroupIds=[
                        os.getenv('AWS_SECURITY_GROUP_ID', 'sg-default')
                    ],
                    DBSubnetGroupName=os.getenv('AWS_DB_SUBNET_GROUP', 'default'),
                    PubliclyAccessible=False,
                    BackupRetentionPeriod=7,
                    StorageEncrypted=True
                )
                
                # Wait for database to be available
                logger.info("Waiting for database to be available...")
                waiter = rds_client.get_waiter('db_instance_available')
                waiter.wait(
                    DBInstanceIdentifier=aws_config["db_instance_id"],
                    WaiterConfig={'Delay': 30, 'MaxAttempts': 60}
                )
                
                # Get endpoint
                response = rds_client.describe_db_instances(
                    DBInstanceIdentifier=aws_config["db_instance_id"]
                )
                return response['DBInstances'][0]['Endpoint']['Address']
                
        except Exception as e:
            logger.error(f"Failed to setup RDS: {e}")
            raise
    
    def _create_ecs_task_definition(self, ecs_client, aws_config: Dict, 
                                   image_uri: str, db_endpoint: str) -> str:
        """Create ECS task definition"""
        logger.info("Creating ECS task definition...")
        
        task_definition = {
            "family": aws_config["task_definition"],
            "networkMode": "awsvpc",
            "requiresCompatibilities": ["FARGATE"],
            "cpu": "256",
            "memory": "512",
            "executionRoleArn": os.getenv('ECS_EXECUTION_ROLE_ARN'),
            "taskRoleArn": os.getenv('ECS_TASK_ROLE_ARN'),
            "containerDefinitions": [
                {
                    "name": "gold-price-api",
                    "image": image_uri,
                    "portMappings": [
                        {
                            "containerPort": 8000,
                            "protocol": "tcp"
                        }
                    ],
                    "environment": [
                        {"name": "ENVIRONMENT", "value": self.environment},
                        {"name": "DATABASE_URL", "value": f"postgresql://golduser:{os.getenv('DB_PASSWORD')}@{db_endpoint}:5432/golddb"},
                        {"name": "REDIS_URL", "value": os.getenv('REDIS_URL', 'redis://localhost:6379')},
                        {"name": "MLFLOW_TRACKING_URI", "value": os.getenv('MLFLOW_TRACKING_URI')}
                    ],
                    "logConfiguration": {
                        "logDriver": "awslogs",
                        "options": {
                            "awslogs-group": f"/ecs/gold-prediction-{self.environment}",
                            "awslogs-region": aws_config["region"],
                            "awslogs-stream-prefix": "ecs"
                        }
                    },
                    "healthCheck": {
                        "command": ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
                        "interval": 30,
                        "timeout": 5,
                        "retries": 3,
                        "startPeriod": 60
                    }
                }
            ]
        }
        
        try:
            response = ecs_client.register_task_definition(**task_definition)
            task_def_arn = response['taskDefinition']['taskDefinitionArn']
            logger.info(f"Task definition created: {task_def_arn}")
            return task_def_arn
            
        except Exception as e:
            logger.error(f"Failed to create task definition: {e}")
            raise
    
    def _create_ecs_service(self, ecs_client, aws_config: Dict, task_def_arn: str) -> str:
        """Create or update ECS service"""
        logger.info("Creating ECS service...")
        
        try:
            # Check if service exists
            try:
                response = ecs_client.describe_services(
                    cluster=aws_config["cluster_name"],
                    services=[aws_config["service_name"]]
                )
                
                if response['services'] and response['services'][0]['status'] == 'ACTIVE':
                    # Update existing service
                    logger.info("Updating existing service...")
                    response = ecs_client.update_service(
                        cluster=aws_config["cluster_name"],
                        service=aws_config["service_name"],
                        taskDefinition=task_def_arn,
                        desiredCount=1
                    )
                    return response['service']['serviceArn']
                    
            except Exception:
                pass
            
            # Create new service
            logger.info("Creating new service...")
            service_definition = {
                "serviceName": aws_config["service_name"],
                "cluster": aws_config["cluster_name"],
                "taskDefinition": task_def_arn,
                "desiredCount": 1,
                "launchType": "FARGATE",
                "networkConfiguration": {
                    "awsvpcConfiguration": {
                        "subnets": os.getenv('AWS_SUBNET_IDS', '').split(','),
                        "securityGroups": [os.getenv('AWS_SECURITY_GROUP_ID', 'sg-default')],
                        "assignPublicIp": "ENABLED"
                    }
                },
                "loadBalancers": [
                    {
                        "targetGroupArn": os.getenv('AWS_TARGET_GROUP_ARN'),
                        "containerName": "gold-price-api",
                        "containerPort": 8000
                    }
                ] if os.getenv('AWS_TARGET_GROUP_ARN') else []
            }
            
            response = ecs_client.create_service(**service_definition)
            service_arn = response['service']['serviceArn']
            logger.info(f"Service created: {service_arn}")
            return service_arn
            
        except Exception as e:
            logger.error(f"Failed to create service: {e}")
            raise
    
    def _wait_for_ecs_deployment(self, ecs_client, aws_config: Dict):
        """Wait for ECS deployment to complete"""
        logger.info("Waiting for deployment to complete...")
        
        try:
            waiter = ecs_client.get_waiter('services_stable')
            waiter.wait(
                cluster=aws_config["cluster_name"],
                services=[aws_config["service_name"]],
                WaiterConfig={'Delay': 15, 'MaxAttempts': 40}
            )
            logger.info("Deployment completed successfully")
            
        except Exception as e:
            logger.error(f"Deployment did not complete: {e}")
            raise
    
    def _get_aws_service_endpoint(self, aws_config: Dict) -> str:
        """Get service endpoint URL"""
        load_balancer_dns = os.getenv('AWS_LOAD_BALANCER_DNS')
        if load_balancer_dns:
            return f"https://{load_balancer_dns}"
        else:
            return f"https://{aws_config['cluster_name']}.{aws_config['region']}.amazonaws.com"
    
    def _deploy_gcp(self, image_tag: str) -> Dict[str, Any]:
        """Deploy to Google Cloud Platform using Cloud Run and Cloud SQL"""
        logger.info("Deploying to GCP...")
        
        try:
            # GCP configuration
            gcp_config = {
                "project_id": os.getenv("GCP_PROJECT_ID"),
                "region": os.getenv("GCP_REGION", "us-central1"),
                "service_name": f"gold-prediction-{self.environment}",
                "image_name": f"gcr.io/{os.getenv('GCP_PROJECT_ID')}/gold-price-prediction",
                "sql_instance": f"gold-db-{self.environment}"
            }
            
            if not gcp_config["project_id"]:
                raise ValueError("GCP_PROJECT_ID environment variable required")
            
            # 1. Push image to Google Container Registry
            gcr_uri = self._push_to_gcr(gcp_config, image_tag)
            
            # 2. Setup Cloud SQL instance
            db_connection = self._setup_gcp_database(gcp_config)
            
            # 3. Deploy to Cloud Run
            service_url = self._deploy_cloud_run(gcp_config, gcr_uri, db_connection)
            
            return {
                "status": "success",
                "provider": "gcp",
                "service": gcp_config["service_name"],
                "endpoint": service_url,
                "image": gcr_uri,
                "database": db_connection
            }
            
        except Exception as e:
            logger.error(f"GCP deployment failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    def _push_to_gcr(self, gcp_config: Dict, image_tag: str) -> str:
        """Push Docker image to Google Container Registry"""
        logger.info("Pushing image to GCR...")
        
        try:
            gcr_uri = f"{gcp_config['image_name']}:{image_tag}"
            
            # Tag image for GCR
            subprocess.run([
                "docker", "tag", image_tag, gcr_uri
            ], check=True)
            
            # Configure Docker for GCR
            subprocess.run([
                "gcloud", "auth", "configure-docker"
            ], check=True)
            
            # Push image
            subprocess.run([
                "docker", "push", gcr_uri
            ], check=True)
            
            logger.info(f"Image pushed to GCR: {gcr_uri}")
            return gcr_uri
            
        except Exception as e:
            logger.error(f"Failed to push to GCR: {e}")
            raise
    
    def _setup_gcp_database(self, gcp_config: Dict) -> str:
        """Setup Cloud SQL PostgreSQL instance"""
        logger.info("Setting up Cloud SQL database...")
        
        try:
            # Use gcloud CLI to manage Cloud SQL
            instance_name = gcp_config["sql_instance"]
            project_id = gcp_config["project_id"]
            
            # Check if instance exists
            result = subprocess.run([
                "gcloud", "sql", "instances", "describe", instance_name,
                "--project", project_id
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                # Create new instance
                logger.info("Creating Cloud SQL instance...")
                subprocess.run([
                    "gcloud", "sql", "instances", "create", instance_name,
                    "--database-version", "POSTGRES_13",
                    "--tier", "db-f1-micro",
                    "--region", gcp_config["region"],
                    "--project", project_id
                ], check=True)
                
                # Create database
                subprocess.run([
                    "gcloud", "sql", "databases", "create", "golddb",
                    "--instance", instance_name,
                    "--project", project_id
                ], check=True)
                
                # Create user
                subprocess.run([
                    "gcloud", "sql", "users", "create", "golduser",
                    "--instance", instance_name,
                    "--password", os.getenv('DB_PASSWORD', 'SecurePass123!'),
                    "--project", project_id
                ], check=True)
            
            # Get connection name
            result = subprocess.run([
                "gcloud", "sql", "instances", "describe", instance_name,
                "--project", project_id,
                "--format", "value(connectionName)"
            ], capture_output=True, text=True, check=True)
            
            connection_name = result.stdout.strip()
            logger.info(f"Cloud SQL connection: {connection_name}")
            return connection_name
            
        except Exception as e:
            logger.error(f"Failed to setup Cloud SQL: {e}")
            raise
    
    def _deploy_cloud_run(self, gcp_config: Dict, image_uri: str, db_connection: str) -> str:
        """Deploy to Cloud Run"""
        logger.info("Deploying to Cloud Run...")
        
        try:
            # Deploy to Cloud Run
            cmd = [
                "gcloud", "run", "deploy", gcp_config["service_name"],
                "--image", image_uri,
                "--platform", "managed",
                "--region", gcp_config["region"],
                "--project", gcp_config["project_id"],
                "--allow-unauthenticated",
                "--port", "8000",
                "--memory", "512Mi",
                "--cpu", "1",
                "--set-env-vars", f"ENVIRONMENT={self.environment}",
                "--set-env-vars", f"DATABASE_URL=postgresql://golduser:{os.getenv('DB_PASSWORD')}@localhost/golddb?host=/cloudsql/{db_connection}",
                "--set-cloudsql-instances", db_connection
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Extract service URL
            import re
            url_match = re.search(r'Service URL: (https://[^\s]+)', result.stdout)
            if url_match:
                service_url = url_match.group(1)
                logger.info(f"Cloud Run service deployed: {service_url}")
                return service_url
            else:
                raise ValueError("Could not extract service URL from deployment output")
                
        except Exception as e:
            logger.error(f"Failed to deploy to Cloud Run: {e}")
            raise
    
    def _deploy_azure(self, image_tag: str) -> Dict[str, Any]:
        """Deploy to Microsoft Azure using Container Instances and Azure Database"""
        logger.info("Deploying to Azure...")
        
        try:
            from azure.identity import DefaultAzureCredential
            from azure.mgmt.containerinstance import ContainerInstanceManagementClient
            from azure.mgmt.sql import SqlManagementClient
        except ImportError:
            raise ImportError("Azure SDK required for Azure deployment. Install with: pip install azure-mgmt-containerinstance azure-mgmt-sql azure-identity")
        
        try:
            # Azure configuration
            azure_config = {
                "subscription_id": os.getenv("AZURE_SUBSCRIPTION_ID"),
                "resource_group": os.getenv("AZURE_RESOURCE_GROUP", f"gold-prediction-{self.environment}"),
                "location": os.getenv("AZURE_LOCATION", "East US"),
                "container_group": f"gold-api-{self.environment}",
                "registry_name": os.getenv("AZURE_CONTAINER_REGISTRY"),
                "sql_server": f"gold-sql-{self.environment}",
                "database_name": "golddb"
            }
            
            if not azure_config["subscription_id"]:
                raise ValueError("AZURE_SUBSCRIPTION_ID environment variable required")
            
            # Initialize clients
            credential = DefaultAzureCredential()
            container_client = ContainerInstanceManagementClient(credential, azure_config["subscription_id"])
            sql_client = SqlManagementClient(credential, azure_config["subscription_id"])
            
            # 1. Push image to Azure Container Registry
            acr_uri = self._push_to_acr(azure_config, image_tag)
            
            # 2. Setup Azure SQL Database
            db_server = self._setup_azure_database(sql_client, azure_config)
            
            # 3. Deploy to Container Instances
            container_url = self._deploy_azure_container(container_client, azure_config, acr_uri, db_server)
            
            return {
                "status": "success",
                "provider": "azure",
                "container_group": azure_config["container_group"],
                "endpoint": container_url,
                "image": acr_uri,
                "database": db_server
            }
            
        except Exception as e:
            logger.error(f"Azure deployment failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    def _push_to_acr(self, azure_config: Dict, image_tag: str) -> str:
        """Push Docker image to Azure Container Registry"""
        logger.info("Pushing image to ACR...")
        
        try:
            registry_name = azure_config["registry_name"]
            if not registry_name:
                raise ValueError("AZURE_CONTAINER_REGISTRY environment variable required")
            
            acr_uri = f"{registry_name}.azurecr.io/gold-price-prediction:{image_tag}"
            
            # Tag image for ACR
            subprocess.run([
                "docker", "tag", image_tag, acr_uri
            ], check=True)
            
            # Login to ACR
            subprocess.run([
                "az", "acr", "login", "--name", registry_name
            ], check=True)
            
            # Push image
            subprocess.run([
                "docker", "push", acr_uri
            ], check=True)
            
            logger.info(f"Image pushed to ACR: {acr_uri}")
            return acr_uri
            
        except Exception as e:
            logger.error(f"Failed to push to ACR: {e}")
            raise
    
    def _setup_azure_database(self, sql_client, azure_config: Dict) -> str:
        """Setup Azure SQL Database"""
        logger.info("Setting up Azure SQL Database...")
        
        try:
            # Use Azure CLI for database setup
            resource_group = azure_config["resource_group"]
            server_name = azure_config["sql_server"]
            location = azure_config["location"]
            
            # Create SQL Server
            subprocess.run([
                "az", "sql", "server", "create",
                "--name", server_name,
                "--resource-group", resource_group,
                "--location", location,
                "--admin-user", "goldadmin",
                "--admin-password", os.getenv('DB_PASSWORD', 'SecurePass123!')
            ], check=True)
            
            # Create database
            subprocess.run([
                "az", "sql", "db", "create",
                "--name", azure_config["database_name"],
                "--server", server_name,
                "--resource-group", resource_group,
                "--service-objective", "Basic"
            ], check=True)
            
            # Configure firewall (allow Azure services)
            subprocess.run([
                "az", "sql", "server", "firewall-rule", "create",
                "--name", "AllowAzureServices",
                "--server", server_name,
                "--resource-group", resource_group,
                "--start-ip-address", "0.0.0.0",
                "--end-ip-address", "0.0.0.0"
            ], check=True)
            
            server_fqdn = f"{server_name}.database.windows.net"
            logger.info(f"Azure SQL Server: {server_fqdn}")
            return server_fqdn
            
        except Exception as e:
            logger.error(f"Failed to setup Azure SQL: {e}")
            raise
    
    def _deploy_azure_container(self, container_client, azure_config: Dict, 
                               image_uri: str, db_server: str) -> str:
        """Deploy to Azure Container Instances"""
        logger.info("Deploying to Azure Container Instances...")
        
        try:
            from azure.mgmt.containerinstance.models import (
                ContainerGroup, Container, ContainerPort, Port, IpAddress,
                EnvironmentVariable, ResourceRequests, ResourceRequirements
            )
            
            # Container configuration
            container = Container(
                name="gold-price-api",
                image=image_uri,
                resources=ResourceRequirements(
                    requests=ResourceRequests(memory_in_gb=0.5, cpu=0.5)
                ),
                ports=[ContainerPort(port=8000)],
                environment_variables=[
                    EnvironmentVariable(name="ENVIRONMENT", value=self.environment),                    EnvironmentVariable(
                        name="DATABASE_URL", 
                        value=f"postgresql://goldadmin:{os.getenv('DB_PASSWORD')}@{db_server}:1433/{azure_config['database_name']}"
                    )
                ]
            )
            
            # Container group configuration
            container_group = ContainerGroup(
                location=azure_config["location"],
                containers=[container],
                os_type="Linux",
                ip_address=IpAddress(
                    type="Public",
                    ports=[Port(protocol="TCP", port=8000)]
                )
            )
            
            # Deploy container group
            result = container_client.container_groups.begin_create_or_update(
                resource_group_name=azure_config["resource_group"],
                container_group_name=azure_config["container_group"],
                container_group=container_group
            ).result()
            
            # Get endpoint URL
            container_url = f"http://{result.ip_address.ip}:8000"
            logger.info(f"Container deployed: {container_url}")
            return container_url
            
        except Exception as e:
            logger.error(f"Failed to deploy to Azure Container Instances: {e}")
            raise


def main():
    """Main function to handle deployment"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Deploy Gold Price Prediction System")
    parser.add_argument("--environment", choices=["development", "staging", "production"], 
                       default="development", help="Target environment")
    parser.add_argument("--provider", choices=["local", "aws", "gcp", "azure"], 
                       default="local", help="Cloud provider")
    parser.add_argument("--build-only", action="store_true", 
                       help="Only build Docker image, don't deploy")
    parser.add_argument("--image-tag", help="Custom Docker image tag")
    
    args = parser.parse_args()
      # Setup logging
    from src.utils import setup_logging
    setup_logging()
    
    try:
        # Initialize deployment manager
        manager = DeploymentManager(args.environment)
        
        # Check prerequisites
        if not manager.check_prerequisites():
            logger.error("Prerequisites check failed. Please resolve issues before deployment.")
            sys.exit(1)
        
        # Build Docker image
        image_tag = manager.build_image(args.image_tag)
        logger.info(f"Built image: {image_tag}")
        
        if args.build_only:
            logger.info("Build-only mode: skipping deployment")
            return
        
        # Deploy based on provider
        if args.provider == "local":
            result = manager.deploy_local(image_tag)
        else:
            result = manager.deploy_cloud(args.provider, image_tag)
        
        # Print deployment results
        if result.get("status") == "success":
            logger.info("Deployment successful!")
            logger.info(f"Deployment details: {result}")
        else:
            logger.error(f"Deployment failed: {result}")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()