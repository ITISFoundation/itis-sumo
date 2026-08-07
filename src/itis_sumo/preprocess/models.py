"""Pydantic models for job/variable selection validation.

De-webbed verbatim from ``mmux_flaskapi.blueprints.dakota_models``
(function-jobs part only; the web API request models stay in mmux/vite).
"""

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def required_completed_jobs(input_vars: list[str], floor: int = 5) -> int:
    """Minimum completed jobs needed to build a Dakota (surfpack) GP surrogate.

    Dakota aborts surrogate construction (opaque "Dakota aborted: Unknown error 250",
    internal MODEL_ERROR) when given <= len(input_vars) training points -- confirmed
    empirically: len(input_vars)+1 points build successfully, len(input_vars) points
    abort. `floor` preserves the historical flat minimum for low-dimensional problems.
    """
    return max(floor, len(input_vars) + 1)


class FunctionJob(BaseModel):
    """Model for a single function job with inputs, outputs, and status."""

    model_config = ConfigDict(
        extra="allow"
    )  # Allow additional fields like job_id, timestamps, etc.

    status: str = Field(
        ..., description="Status of the job (e.g., 'completed', 'success', 'failed')"
    )
    inputs: dict[str, float | int] = Field(..., description="Input parameters (key-number pairs)")
    outputs: dict[str, float | int] = Field(..., description="Output results (key-number pairs)")

    @field_validator("status")
    @classmethod
    def status_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Status cannot be empty")
        return v.strip().lower()

    @field_validator("inputs")
    @classmethod
    def inputs_must_have_values(
        cls, v: dict[str, float | int | str]
    ) -> dict[str, float | int | str]:
        if not v:
            raise ValueError("Inputs dictionary cannot be empty")
        return v

    @field_validator("outputs")
    @classmethod
    def outputs_must_have_values(
        cls, v: dict[str, float | int | str]
    ) -> dict[str, float | int | str]:
        if not v:
            raise ValueError("Outputs dictionary cannot be empty")
        return v


class JobVariableSelection(BaseModel):
    """Validated selection of jobs and variables for workflow helpers."""

    jobs: list[FunctionJob] = Field(..., min_length=1)
    input_vars: list[str] = Field(..., min_length=1)
    output_vars: list[str] = Field(..., min_length=1)
    minimum_completed_jobs: int = Field(5, ge=1)

    @field_validator("input_vars", "output_vars")
    @classmethod
    def variable_names_must_not_be_empty_strings(cls, v: list[str]) -> list[str]:
        cleaned = []
        for var in v:
            if not var or not var.strip():
                raise ValueError("Variable names cannot be empty")
            cleaned.append(var.strip())
        return cleaned

    @property
    def completed_jobs(self) -> list[FunctionJob]:
        return [job for job in self.jobs if job.status in ["completed", "success"]]

    @model_validator(mode="after")
    def validate_completed_jobs_have_requested_variables(self) -> "JobVariableSelection":
        completed_jobs = self.completed_jobs

        if len(completed_jobs) < self.minimum_completed_jobs:
            raise ValueError(
                "At least "
                f"{self.minimum_completed_jobs} samples are necessary to build a surrogate model in Dakota "
                f"(dimension-scaled minimum: max(5, num_input_vars + 1) = "
                f"max(5, {len(self.input_vars)} + 1)). "
                f"Found {len(completed_jobs)} completed jobs."
            )

        missing_input_vars = set()
        missing_output_vars = set()
        available_input_keys = set()
        available_output_keys = set()

        for job in completed_jobs:
            available_input_keys.update(job.inputs.keys())
            available_output_keys.update(job.outputs.keys())

            for input_var in self.input_vars:
                if input_var not in job.inputs:
                    missing_input_vars.add(input_var)

            for output_var in self.output_vars:
                if output_var not in job.outputs:
                    missing_output_vars.add(output_var)

        if missing_input_vars:
            raise ValueError(
                f"Input variables {sorted(missing_input_vars)} not found in completed job inputs. "
                f"Available input keys: {sorted(available_input_keys)}"
            )

        if missing_output_vars:
            raise ValueError(
                f"Output variables {sorted(missing_output_vars)} not found in completed job outputs. "
                f"Available output keys: {sorted(available_output_keys)}"
            )

        return self

    def to_records(self) -> list[dict[str, float | int]]:
        records = []
        for job in self.completed_jobs:
            record: dict[str, float | int] = {}
            for input_var in self.input_vars:
                record[input_var] = job.inputs[input_var]
            for output_var in self.output_vars:
                record[output_var] = job.outputs[output_var]
            records.append(record)
        return records
