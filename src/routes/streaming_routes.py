from flask import Blueprint
from src.controllers.streamingControllers.streaming_output_controller import StreamingOutput

streaming_blueprint = Blueprint('streaming_blueprint', __name__)

@streaming_blueprint.route('/streaming_output/<client>/<project>/<stream>/<username>/<scenario>/<number_of_rows>', methods=['GET'])
def get_streaming_output(client, project, stream, username, scenario, number_of_rows):
    Streaming_output = StreamingOutput(client, project, stream, username, scenario, number_of_rows)
    return Streaming_output.get_streaming_output()
