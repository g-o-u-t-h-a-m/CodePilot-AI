from pydantic import BaseModel, Field, HttpUrl


class CloneRepositoryRequest(BaseModel):
    """Request model for cloning a repository."""
    github_url: str = Field(
        ...,
        description="GitHub repository URL",
        examples=["https://github.com/facebook/react"]
    )


class CloneRepositoryResponse(BaseModel):
    """Response model for repository clone operation."""
    success: bool = Field(..., description="Whether the operation was successful")
    repository_name: str = Field(..., description="Name of the cloned repository")
    local_path: str = Field(..., description="Local path where repository is stored")
    message: str = Field(..., description="Status message")
