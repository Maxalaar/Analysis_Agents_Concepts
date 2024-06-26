def flattening(dictionary: dict, flatten_dictionary: dict = {}, prefix: str = ''):
    for key, value in dictionary.items():
        if type(value) is dict:
            flattening(
                dictionary=value,
                flatten_dictionary=flatten_dictionary,
                prefix=prefix+'_'+str(key)
            )
        else:
            if value is not None:
                flatten_dictionary[prefix+'_'+str(key)] = value

    return flatten_dictionary
