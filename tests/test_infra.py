"""Infrastructure configuration tests for Kubernetes manifests and Dockerfile.

Ops Lead: Prakhar Gupta (S24CSEU1006)
"""

from pathlib import Path
import yaml


def test_k8s_deployment_configuration():
    """Verify k8s/deployment.yaml has exact name, image, replicas, ports, and probes."""
    deploy_path = Path("k8s/deployment.yaml")
    assert deploy_path.exists()
    with open(deploy_path) as f:
        deploy = yaml.safe_load(f)

    assert deploy["metadata"]["name"] == "mp266-api"
    assert deploy["spec"]["replicas"] == 4

    container = deploy["spec"]["template"]["spec"]["containers"][0]
    assert container["name"] == "mp266-api"
    assert container["image"] == "mp266-api:1.0"
    assert any(p["containerPort"] == 8000 for p in container["ports"])

    # Readiness & Liveness Probes
    assert "readinessProbe" in container
    assert "livenessProbe" in container
    assert container["readinessProbe"]["httpGet"]["path"] == "/health"
    assert container["livenessProbe"]["httpGet"]["path"] == "/health"
    assert container["livenessProbe"]["initialDelaySeconds"] > container["readinessProbe"]["initialDelaySeconds"]


def test_k8s_service_configuration():
    """Verify k8s/service.yaml has exact name, matching selector, and port forwarding."""
    svc_path = Path("k8s/service.yaml")
    assert svc_path.exists()
    with open(svc_path) as f:
        svc = yaml.safe_load(f)

    assert svc["metadata"]["name"] == "mp266-svc"
    assert svc["spec"]["selector"]["app"] == "mp266-api"

    port_cfg = svc["spec"]["ports"][0]
    assert port_cfg["port"] == 80
    assert port_cfg["targetPort"] == 8000


def test_dockerfile_and_dockerignore():
    """Verify Dockerfile and .dockerignore exist and are correctly structured."""
    di = Path(".dockerignore")
    assert di.exists()
    di_content = di.read_text()
    assert ".git" in di_content
    assert ".venv" in di_content

    df = Path("Dockerfile")
    if df.exists():
        content = df.read_text()
        assert "FROM python:3.12-slim" in content
        assert "EXPOSE 8000" in content
        assert "uvicorn" in content
