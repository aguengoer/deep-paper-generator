# This is a sample Python script.
import DataProcessor
from PipelineLemming import PipelineLemming
from PipelineStemming import PipelineStemming

if __name__ == '__main__':
    data = DataProcessor.process()

    #lemming
    lemming = PipelineLemming()
    lemming_processed_data = lemming.process(data)

    #stemming
    stemming = PipelineStemming()
    stemming_processed_data = stemming.process(lemming_processed_data)


