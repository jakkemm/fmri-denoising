def iter_chunks(Y, chunk_size):
    for start in range(0, Y.shape[1], chunk_size):
        stop = min(start + chunk_size, Y.shape[1])

        chunk_slice = slice(start, stop)
        yield chunk_slice, Y[:, chunk_slice]