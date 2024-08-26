def clip(number, _min, _max):
    if number <= _min:
        return _min
    elif number >= _max:
        return _max
    else:
        return number