from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    database_url: str = "sqlite:///./cdss.db"
    llm_api_base: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    crawler_data_path: str = "../crawler"
    guideline_data_path: str = "../guideline"
    max_concurrent_extractions: int = 5
    training_runs_dir: str = "./training_runs"
    pretrain_runs_dir: str = "./pretrain_runs"

    class Config:
        env_file = ".env"

settings = Settings()
