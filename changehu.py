import os
from transformers import AutoModel

# Set the new cache directory
print(os.getenv('TRANSFORMERS_CACHE')) 