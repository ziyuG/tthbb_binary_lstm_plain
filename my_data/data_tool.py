
def even_odd_split(arr, eventNumber):
    arr_even = arr[eventNumber % 2 == 0]
    arr_odd = arr[eventNumber % 2 == 1]

    return arr_even, arr_odd

