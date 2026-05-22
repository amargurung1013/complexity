from collections import Counter


def bubble_sort(values: list[int]) -> None:
    for end in range(len(values) - 1, 0, -1):
        swapped = False
        for index in range(end):
            if values[index] > values[index + 1]:
                values[index], values[index + 1] = values[index + 1], values[index]
                swapped = True
        if not swapped:
            break


def selection_sort(values: list[int]) -> None:
    for index in range(len(values)):
        min_index = index
        for candidate in range(index + 1, len(values)):
            if values[candidate] < values[min_index]:
                min_index = candidate
        values[index], values[min_index] = values[min_index], values[index]


def insertion_sort(values: list[int]) -> None:
    for index in range(1, len(values)):
        current = values[index]
        previous = index - 1
        while previous >= 0 and values[previous] > current:
            values[previous + 1] = values[previous]
            previous -= 1
        values[previous + 1] = current


def merge_sort(values: list[int]) -> None:
    if len(values) <= 1:
        return

    midpoint = len(values) // 2
    left = values[:midpoint]
    right = values[midpoint:]

    merge_sort(left)
    merge_sort(right)

    left_index = 0
    right_index = 0
    write_index = 0

    while left_index < len(left) and right_index < len(right):
        if left[left_index] <= right[right_index]:
            values[write_index] = left[left_index]
            left_index += 1
        else:
            values[write_index] = right[right_index]
            right_index += 1
        write_index += 1

    while left_index < len(left):
        values[write_index] = left[left_index]
        left_index += 1
        write_index += 1

    while right_index < len(right):
        values[write_index] = right[right_index]
        right_index += 1
        write_index += 1


def quick_sort(values: list[int]) -> None:
    def partition(low: int, high: int) -> int:
        pivot = values[high]
        boundary = low - 1

        for index in range(low, high):
            if values[index] <= pivot:
                boundary += 1
                values[boundary], values[index] = values[index], values[boundary]

        values[boundary + 1], values[high] = values[high], values[boundary + 1]
        return boundary + 1

    stack = [(0, len(values) - 1)]
    while stack:
        low, high = stack.pop()
        if low < high:
            pivot_index = partition(low, high)
            stack.append((low, pivot_index - 1))
            stack.append((pivot_index + 1, high))


def heap_sort(values: list[int]) -> None:
    def heapify(heap_size: int, root_index: int) -> None:
        largest = root_index
        left = (2 * root_index) + 1
        right = (2 * root_index) + 2

        if left < heap_size and values[left] > values[largest]:
            largest = left
        if right < heap_size and values[right] > values[largest]:
            largest = right
        if largest != root_index:
            values[root_index], values[largest] = values[largest], values[root_index]
            heapify(heap_size, largest)

    for index in range((len(values) // 2) - 1, -1, -1):
        heapify(len(values), index)

    for index in range(len(values) - 1, 0, -1):
        values[index], values[0] = values[0], values[index]
        heapify(index, 0)


def counting_sort(values: list[int]) -> None:
    if not values:
        return

    counts = Counter(values)
    write_index = 0

    for value in range(min(counts), max(counts) + 1):
        for _ in range(counts[value]):
            values[write_index] = value
            write_index += 1


def tim_sort(values: list[int]) -> None:
    values.sort()
