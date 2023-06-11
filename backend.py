from jina import Flow, Document
import config


if __name__ == "__main__":

    # create Flow
    flow = Flow(host=config.HOST, protocol=config.PROTOCOL,
                port_expose=config.PORT).add(uses='jinahub://PetBreedClassifier')
    with flow:
        flow.block()
