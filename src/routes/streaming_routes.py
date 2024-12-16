from flask import Blueprint
from src.controllers.streamingControllers.streaming_output_controller import StreamingOutput

streaming_Blueprint = Blueprint('streaming_Blueprint', __name__, url_prefix='/streaming')

@streaming_Blueprint.route('/streaming_output/<client>/<project>/<stream>/<username>/<scenario>/<number_of_rows>', methods=['GET'])
def get_all_devices(client, project, stream, username, scenario, number_of_rows):
    streaming_output_controller = StreamingOutput(client, project, stream, username, scenario, number_of_rows)
    return streaming_output_controller.get_streaming_output()