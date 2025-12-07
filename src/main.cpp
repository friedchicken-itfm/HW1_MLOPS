#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>

namespace py = pybind11;

// MAC для 3D тензоров: (Batch, N, K) * (Batch, K, M) + (Batch, N, M)
// возвращает тензор
py::array_t<float> tensor_mac(py::array_t<float> input_a, py::array_t<float> input_b, py::array_t<float> input_c) {
    
    //получаем буфео
    auto buf_a = input_a.request();
    auto buf_b = input_b.request();
    auto buf_c = input_c.request();

    // проверка размерностей
    if (buf_a.ndim != 3 || buf_b.ndim != 3 || buf_c.ndim != 3)
        throw std::runtime_error("ошибка размерности");

    // размеры
    const int batch = buf_a.shape[0];
    const int N = buf_a.shape[1];
    const int K = buf_a.shape[2];
    const int M = buf_b.shape[2];

    if (buf_b.shape[0] != batch || buf_b.shape[1] != K)
         throw std::runtime_error("несовпадение размеров");
    
    // результат
    auto result = py::array_t<float>({batch, N, M});
    auto buf_res = result.request();

    float* ptr_a = static_cast<float*>(buf_a.ptr);
    float* ptr_b = static_cast<float*>(buf_b.ptr);
    float* ptr_c = static_cast<float*>(buf_c.ptr);
    float* ptr_res = static_cast<float*>(buf_res.ptr);

    // наивная реализация
    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < M; ++j) {
                float sum = 0.0f;
                for (int p = 0; p < K; ++p) {
                    //b*N*K + i*K + p
                    sum += ptr_a[b * N * K + i * K + p] * ptr_b[b * K * M + p * M + j];
                }
                // добавляем с
                ptr_res[b * N * M + i * M + j] = sum + ptr_c[b * N * M + i * M + j];
            }
        }
    }

    return result;
}

PYBIND11_MODULE(hw1_mac, m) {
    m.doc() = "HW1 MLOps: реализация MAC на C++";
    m.def("tensor_mac", &tensor_mac, "посчитать A*B +C ");
}