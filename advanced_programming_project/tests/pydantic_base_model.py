from pydantic import BaseModel, HttpUrl, DirectoryPath, validator
from typing import List

class FlightDataAnalysisConfig(BaseModel):
    data_dir: DirectoryPath
    file_name: str
    processed_data_dir: DirectoryPath
    download_url: HttpUrl
    dataset_names: List[str]

    @validator('file_name')
    def validate_file_name(cls, v):
        if not v.endswith('.zip'):
            raise ValueError('file_name must be a .zip file')
        return v

    @validator('dataset_names')
    def validate_dataset_names(cls, v):
        if not v:
            raise ValueError('dataset_names cannot be empty')
        if not all(isinstance(name, str) for name in v):
            raise ValueError('All dataset names must be strings')
        return v
