from abc import *

class DataBasket(metaclass=ABCMeta):
    @abstractmethod
    def get_date(self, file_address):
        pass

    @abstractmethod
    def split_data(self, X, Y):
        pass





