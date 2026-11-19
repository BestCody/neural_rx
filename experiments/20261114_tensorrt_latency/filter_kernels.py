"""Host side of fitted_filter.cu: uploads the tables and one slot's inputs, and launches the kernels.

    kernels = FilterKernels(np, cuda, cudart, cubin, tables, feeds, output, stream)
    kernels.fit()       # before the slot: weights from the memory, and the memory's part of the predictions
    kernels.apply()     # on the slot: predictions, then choice of the entries and refined channel into `output`

`tables` is build/filter_tables.npz and `feeds` the "filter/input/..." arrays of build/reference.npz (export.py);
`output` is the device address of the network's channel input, [U, positions, 2B].
"""
from time_trt import call


class FilterKernels:
    FIELDS = ("ls", "noise", "memory", "age", "ports", "valid", "active", "pack", "noise_factors", "neighbor_index",
              "neighbor_valid", "neighbor_delta", "cell_symbol", "symbol_cells", "slope_cells", "cell_of_position",
              "time_factor", "frequency_factor", "pair_row", "pair_column", "gram", "rhs", "weights", "available",
              "memory_part", "prediction", "sums", "choice", "gain", "out")

    def __init__(self, np, cuda, cudart, cubin, tables, feeds, output, stream, group_cells=16, expand_threads=256):
        self.np, self.cuda, self.cudart = np, cuda, cudart
        call(cuda.cuInit(0))
        self.module = call(cuda.cuModuleLoadData(cubin))
        self.functions = {name: call(cuda.cuModuleGetFunction(self.module, name.encode()))
                          for name in ("filter_gram", "filter_solve", "filter_history", "filter_predict",
                                       "filter_expand")}
        users, cells = tables["pack"].shape
        positions = tables["cell_of_position"].shape[1]
        memory = feeds["memory_observations"]
        history, antennas = memory.shape[2], (memory.shape[-1]-1)//2
        neighbors, symbols, symbol_cells = tables["neighbor_index"].shape[-1], *tables["symbol_cells"].shape[1:]
        fit_tbs = int(tables["fit_tbs"])
        entries = history-fit_tbs
        inputs = (entries+1)*neighbors
        groups = -(-symbol_cells//group_cells)      # filter_predict sums over groups of cells of one DMRS symbol
        rows, columns = np.triu_indices(inputs)
        if (memory.shape != (1, users, history, cells, 2*antennas+1) or feeds["h_hat_ls"].shape[1] != users
                or np.prod(feeds["h_hat_ls"].shape[2:4]) != positions or feeds["h_hat_ls"].shape[-1] != 2*antennas
                or fit_tbs > 16 or entries < 0):
            raise ValueError("tables and inputs do not match")
        f32, i32 = np.float32, np.int32
        self.sizes = dict(users=users, cells=cells, positions=positions, antennas=antennas, history=history,
                          neighbors=neighbors, symbols=symbols, symbol_cells_count=symbol_cells, entries=entries,
                          fit_tbs=fit_tbs, pairs=len(rows), groups=groups)
        self.shapes = {"gram": (users, symbols, fit_tbs, len(rows), 2), "rhs": (users, symbols, fit_tbs, inputs, 2),
                       "weights": (users, symbols, entries+1, inputs, 2), "available": (users, symbols, entries+1),
                       "memory_part": (users, cells, entries+1, 2*antennas),
                       "prediction": (users, cells, entries+1, 2*antennas), "sums": (users, symbols, groups, entries+2),
                       "choice": (users, symbols), "gain": (users, symbols)}
        self.types = {"available": i32, "choice": i32}
        self.pointers = {
            "ls": self._upload(feeds["h_hat_ls"], f32), "noise": self._upload(feeds["noise_variance"], f32),
            "memory": self._upload(memory, f32), "age": self._upload(feeds["memory_age"], i32),
            "ports": self._upload(feeds["memory_ports"], i32), "valid": self._upload(feeds["memory_valid"], f32),
            "active": self._upload(feeds["active"], f32), "pack": self._upload(tables["pack"], i32),
            "noise_factors": self._upload(tables["noise_factors"], f32),
            "neighbor_index": self._upload(tables["neighbor_index"], i32),
            "neighbor_valid": self._upload(tables["neighbor_valid"], np.uint8),
            "neighbor_delta": self._upload(tables["neighbor_delta"], f32),
            "cell_symbol": self._upload(tables["cell_symbol"], i32),
            "symbol_cells": self._upload(tables["symbol_cells"], i32),
            "slope_cells": self._upload(tables["slope_cells"], i32),
            "cell_of_position": self._upload(tables["cell_of_position"], i32),
            "time_factor": self._upload(tables["time_factor"], f32),
            "frequency_factor": self._upload(tables["frequency_factor"], f32),
            "pair_row": self._upload(rows, i32), "pair_column": self._upload(columns, i32), "out": output}
        for name, shape in self.shapes.items():
            self.pointers[name] = self._upload(np.zeros(shape, self.types.get(name, f32)), self.types.get(name, f32))
        # the kernels' context: every field is 8 bytes
        words = np.zeros(len(self.FIELDS)+len(self.sizes)+1, np.uint64)
        words[:len(self.FIELDS)] = [int(self.pointers[name]) for name in self.FIELDS]
        words[len(self.FIELDS):-1] = np.array(list(self.sizes.values()), np.int64).view(np.uint64)
        words[-1:] = np.array([float(tables["ridge"])], np.float64).view(np.uint64)
        self.context = words                        # passed to the kernels by value
        self.params = np.array([self.context.ctypes.data], np.uint64)
        self.stream = cuda.CUstream(int(stream))
        solver = 1 << int(np.ceil(np.log2(inputs+1)))
        self.launches = {
            "filter_gram": ((len(rows)+inputs, users*symbols*fit_tbs, 1), (128, 1, 1), 128*8),
            "filter_solve": ((entries+1, users*symbols, 1), (solver, 1, 1), (inputs*(inputs+1)+2*inputs+1)*8),
            "filter_history": ((cells, users, 1), (neighbors, entries+1, 1), inputs*antennas*8),
            "filter_predict": ((groups, users*symbols, 1), (antennas, group_cells, 1),
                               antennas*group_cells*(neighbors*8+(entries+2)*4)),
            "filter_expand": ((-(-positions//expand_threads), users, 1), (expand_threads, 1, 1),
                              symbols*(entries+4)*4)}

    def _upload(self, array, dtype):
        array = self.np.ascontiguousarray(array, dtype)
        pointer = call(self.cudart.cudaMalloc(max(array.nbytes, 8)))
        call(self.cudart.cudaMemcpy(pointer, array.ctypes.data, array.nbytes,
                                    self.cudart.cudaMemcpyKind.cudaMemcpyHostToDevice))
        return pointer

    def _launch(self, name):
        grid, block, shared = self.launches[name]
        call(self.cuda.cuLaunchKernel(self.functions[name], *grid, *block, shared, self.stream,
                                      self.params.ctypes.data, 0))

    def fit(self):
        self._launch("filter_gram")
        self._launch("filter_solve")
        self._launch("filter_history")

    def apply(self):
        self._launch("filter_predict")
        self._launch("filter_expand")

    def read(self, name):
        """A result buffer as a host array (after the stream has been synchronized)."""
        host = self.np.empty(self.shapes[name], self.types.get(name, self.np.float32))
        call(self.cudart.cudaMemcpy(host.ctypes.data, self.pointers[name], host.nbytes,
                                    self.cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost))
        return host
