from fileformats.application import Json
from pydra.tasks.common import LoadJson

# Create a sample JSON file to test
json_file = Json.sample()

# Parameterise the task to load the JSON file
load_json = LoadJson(file=json_file)

# Run the task
outputs = load_json(plugin="serial")

# Print the output interface of the of the task (LoadJson.Outputs)
print(outputs)
