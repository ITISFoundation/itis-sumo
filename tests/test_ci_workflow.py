from pathlib import Path

WORKFLOWS_DIR = Path(__file__).parents[1] / ".github" / "workflows"
CI_WORKFLOW = WORKFLOWS_DIR / "ci.yml"


def test_v18_workflow_changes_run_validation_matrix():
    workflow = CI_WORKFLOW.read_text()

    assert "ci: ${{ steps.filter.outputs.ci }}" in workflow
    assert ".github/workflows/*)" in workflow
    assert workflow.count("needs.detect_changes.outputs.ci == 'true'") == 4


def test_setup_uv_uses_published_tag():
    setup_uv_references = [
        line.strip()
        for workflow_path in WORKFLOWS_DIR.glob("*.yml")
        for line in workflow_path.read_text().splitlines()
        if "astral-sh/setup-uv@" in line
    ]

    assert setup_uv_references
    assert all(reference.endswith("@v9.0.0") for reference in setup_uv_references)
