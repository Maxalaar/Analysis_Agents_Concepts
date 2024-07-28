import dask.array as da


def custom_scaling(column):
    largest_negative_value = da.nanmin(column)
    largest_positive_value = da.nanmax(column)

    norm_of_largest_negative = da.abs(largest_negative_value)

    column = da.where(column < 0, column / norm_of_largest_negative, column)
    column = da.where(column >= 0, column / largest_positive_value, column)

    return column