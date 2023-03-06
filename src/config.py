
from pydantic import BaseSettings, Field

class BaseConfig(BaseSettings):
    """Define any config here.

    See here for documentation:
    https://pydantic-docs.helpmanual.io/usage/settings/
    """
    # KNative assigns a $PORT environment variable to the container
    port: int = Field(default=8084, env="PORT",description="Gradio App Server Port")

    spec_model_path: str = 'models/tts_en_fastpitch_align.nemo'
    vocoder_model_path: str = 'models/tts_hifigan.nemo'

    sample_rate: int = 22050

config = BaseConfig()