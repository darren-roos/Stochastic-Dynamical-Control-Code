
def zeros(rows, cols, fun=None):
    fun = (lambda : 0) if fun==None else fun
    return [[fun() for col in range(cols)] for row in range(rows)]
