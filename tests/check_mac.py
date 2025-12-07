import numpy as np
import hw1_mac
import time

def test_binding():
    print("запуск теста реализации")
    
    # параметры тензоров
    Batch, N, K, M = 10, 32, 64, 32
    
    #генерим тензоры
    np.random.seed(42)
    A = np.random.rand(Batch, N, K).astype(np.float32)
    B = np.random.rand(Batch, K, M).astype(np.float32)
    C = np.random.rand(Batch, N, M).astype(np.float32)

    # эталон
    expected = (A @ B) + C
    
    # наша реализация на C++
    start_time = time.time()
    result = hw1_mac.tensor_mac(A, B, C)
    end_time = time.time()
    
    print(f"C++ implementation time: {end_time - start_time:.6f} sec")

    
    #при сравнении используем allclose ,тк float32 имеет погрешностипри округлении
    if np.allclose(result, expected, atol=1e-5):
        print("SUCCESS :)")
    else:
        print("FAILED :(")
        diff = np.abs(result - expected)
        print(f"Max difference: {np.max(diff)}")
        exit(1)

if __name__ == "__main__":
    test_binding()