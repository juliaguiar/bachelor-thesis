class Violence(object):
  def __init__(self, value, type):
    self.__value = value
    self.__type = type

  def __repr__(self):
    return "value:%s type:%s" % (self.__value, self.__type)

  def get_value(self):
    return self.__value

  def get_type(self):
    return self.__type
