#include <stdlib.h>
#include <cmath>
#include <iostream>
#include <limits>
#include <algorithm>
#include <memory>


#include <immintrin.h>
#include <x86intrin.h>

template <typename T>
struct mem_traits {
    static constexpr size_t padding   = 768;
    static constexpr size_t alignment = 32;

    static constexpr size_t array_padding = 2 * (alignment / sizeof(T)) + 1;
};

template<>
struct mem_traits<double> {
    static constexpr size_t padding   = 768;
    static constexpr size_t alignment = 32;
    static constexpr size_t array_padding = 2 * (alignment / sizeof(double)) + 1;
};

template<>
struct mem_traits<float> {
    static constexpr size_t padding   = 384;
    static constexpr size_t alignment = 32;
    static constexpr size_t array_padding = 2 * (alignment / sizeof(float)) + 1;
};
////////////////////////////////////////////////////////////////////////////////
// Vectorized Functions
////////////////////////////////////////////////////////////////////////////////

typedef float  v8f __attribute__ ((vector_size (32)));
typedef double v4d __attribute__ ((vector_size (32)));

enum class InstructionSet {AVX2};

template<typename TFloat, InstructionSet = InstructionSet::AVX2>
TFloat scaled_dot_product_primitive(TFloat *y_1, TFloat *y_2, TFloat *s);

template <>
double scaled_dot_product_primitive(double *y_1, double *y_2, double *s)
{
    v4d *y_1_v = reinterpret_cast<v4d*>(y_1);
    v4d *y_2_v = reinterpret_cast<v4d*>(y_2);
    v4d *s_v   = reinterpret_cast<v4d*>(s);
    v4d dy, dy_p;
    dy   = __builtin_ia32_subpd256(*y_1_v, *y_2_v);
    dy   = __builtin_ia32_mulpd256(dy, dy);
    dy   = __builtin_ia32_mulpd256(dy, *s_v);
    dy_p = __builtin_ia32_vperm2f128_pd256(dy, dy, 1);
    dy   = __builtin_ia32_addpd256(dy_p, dy);
    dy   = __builtin_ia32_haddpd256(dy, dy);
    return reinterpret_cast<double*>(&dy)[0];
}

template <>
float scaled_dot_product_primitive(float *y_1, float *y_2, float *s)
{
    v8f *y_1_v = reinterpret_cast<v8f*>(y_1);
    v8f *y_2_v = reinterpret_cast<v8f*>(y_2);
    v8f *s_v   = reinterpret_cast<v8f*>(s);
    v8f dy, dy_p;
    dy   = __builtin_ia32_subps256(*y_1_v, *y_2_v);
    dy   = __builtin_ia32_mulps256(dy, dy);
    dy   = __builtin_ia32_mulps256(dy, *s_v);
    dy_p = __builtin_ia32_vperm2f128_ps256(dy, dy, 1);
    dy   = __builtin_ia32_addps256(dy_p, dy);
    dy   = __builtin_ia32_haddps256(dy, dy);
    dy   = __builtin_ia32_haddps256(dy, dy);
    return reinterpret_cast<float*>(&dy)[0];
}

template<typename TFloat, size_t l>
TFloat scaled_dot_product(TFloat *y_1, TFloat *y_2, TFloat *s)
{
    constexpr size_t block_size = 32 / sizeof(TFloat);

    TFloat result = 0.0;
    for (size_t i = 0; i < l; i+=block_size) {
        result = scaled_dot_product_primitive<TFloat>(y_1, y_2, s);
    }
    return result;
}

template<typename TFloat, InstructionSet = InstructionSet::AVX2>
TFloat exponential(TFloat *x);

template<>
float exponential<float, InstructionSet::AVX2>(float *x)
{
    v8f *x_v = reinterpret_cast<v8f*>(x);
    x_v = _mm256_exp_ps(x);
}

template<>
double exponential<double, InstructionSet::AVX2>(double *x)
{
    v4d *x_v = reinterpret_cast<v4d*>(x);
    x_v = _mm256_exp_pd(x);
}

////////////////////////////////////////////////////////////////////////////////
// Aligned Array
////////////////////////////////////////////////////////////////////////////////
template <typename TFloat, size_t alignment>
class AlignedArray {
public:
    AlignedArray() = default;
    AlignedArray(size_t size)
        : data_(nullptr), data_ptr_(nullptr)
    {
        size_t space = size + alignment / sizeof(TFloat) + 1;
        data_ptr_ = std::make_unique<TFloat[]>(space);
        data_ = data_ptr_.get();
        void *ptr = static_cast<void*>(data_);
        if (std::align(alignment, size, ptr, space)) {
            data_ = static_cast<TFloat*>(ptr);
        } else {
            throw std::runtime_error("Couldn't alig storage.");
        }
    }
    AlignedArray(const AlignedArray &)  = delete;
    AlignedArray(      AlignedArray &&) = default;
    AlignedArray & operator=(const AlignedArray &)  = delete;
    AlignedArray & operator=(      AlignedArray &&) = default;
    ~AlignedArray() = default;

    TFloat & operator[](size_t i )       {return data_[i];}
    TFloat   operator[](size_t i ) const {return data_[i];}

private:
    std::unique_ptr<TFloat[]> data_ptr_;
    TFloat *data_;
};

////////////////////////////////////////////////////////////////////////////////
// Padded Matrix
////////////////////////////////////////////////////////////////////////////////
template <typename TFloat, size_t row_block_size, size_t col_block_size, size_t alignment>
class PaddedMatrix {
public:
    PaddedMatrix(size_t m, size_t n)
        : m_(m), n_(n), data_()
    {
        n_row_blocks_ = (n_ / row_block_size);
        if ((n % row_block_size) != 0) {
            ++n_row_blocks_;
        }
        n_col_blocks_ = (m_ / col_block_size);
        if ((m % col_block_size) != 0) {
            ++n_col_blocks_;
        }
        size_t size = n_row_blocks_ * row_block_size * n_col_blocks_ * col_block_size;
        data_ = AlignedArray<TFloat, alignment>(size);
        row_length_ = n_row_blocks_ * row_block_size;
    }
    PaddedMatrix(const PaddedMatrix &)  = delete;
    PaddedMatrix(      PaddedMatrix &&) = default;
    PaddedMatrix & operator=(const PaddedMatrix &)  = delete;
    PaddedMatrix & operator=(      PaddedMatrix &&) = default;
    ~PaddedMatrix() = default;

    TFloat & operator()(size_t i, size_t j)       {return data_[i * row_length_ + j];}
    TFloat   operator()(size_t i, size_t j) const {return data_[i * row_length_ + j];}

    template<typename T>
    void copy(const T *src) {
        for (size_t i = 0; i < n_col_blocks_ * col_block_size; ++i) {
            for (size_t j = 0; j < n_row_blocks_ * row_block_size; ++j) {
                if ((i < m_) && (j < n_)) {
                    this->operator()(i,j) = src[i * n_ + j];
                } else {
                    this->operator()(i,j) = 0.0;
                }
            }
        }
    }

private:
    size_t m_, n_, n_row_blocks_, n_col_blocks_, row_length_;
    AlignedArray<TFloat, alignment> data_;
};

////////////////////////////////////////////////////////////////////////////////
// Padded Vector
////////////////////////////////////////////////////////////////////////////////
template <typename TFloat, size_t block_size, size_t alignment>
struct PaddedVector {
public:
    PaddedVector(size_t m)
        : m_(m), data_()
    {
        n_blocks_ = (m / block_size);
        if ((m_ % block_size) != 0) {
            ++n_blocks_;
        }
        size_t size = n_blocks_ * block_size;
        data_ = AlignedArray<TFloat, alignment>(size);
    }
    PaddedVector(const PaddedVector &)  = delete;
    PaddedVector(      PaddedVector &&) = default;
    PaddedVector & operator=(const PaddedVector &)  = delete;
    PaddedVector & operator=(      PaddedVector &&) = delete;
    ~PaddedVector() = default;

    TFloat & operator[](size_t i)       {return data_[i];}
    TFloat   operator[](size_t i) const {return data_[i];}

    template<typename T>
    void copy(const T *src) {
        for (size_t i = 0; i < n_blocks_ * block_size; ++i) {
            if (i < m_) {
                data_[i] = src[i];
            } else {
                data_[i] = 0.0;
            }
        }
    }

private:
    size_t m_, n_blocks_;
    AlignedArray<TFloat, alignment> data_;
};

////////////////////////////////////////////////////////////////////////////////
// The BMCI Class
////////////////////////////////////////////////////////////////////////////////
template <typename TFloat, template<typename> class TMem = mem_traits>
class BMCI {

    // Alignment Properties
    static constexpr size_t alignment     = TMem<TFloat>::alignment;
    static constexpr size_t array_padding = TMem<TFloat>::array_padding;
    static constexpr size_t row_padding   = TMem<TFloat>::padding / sizeof(TFloat);

public:
    BMCI(const double *Y, const double *x, const double *s, size_t m, size_t n)
        : m_(m), n_(n), hist_min_(std::numeric_limits<TFloat>::max()), hist_max_(0.0), Y_(m,n), x_(m), s_inv_(n)
    {
        // Copy Data
        Y_.copy(Y);
        s_inv_.copy(s);
        x_.copy(x);

        for (size_t i = 0; i < n; ++i) {
            s_inv_[i] = static_cast<TFloat>(1.0 / s[i]);
        }

        // Get Histogram Limits.
        for (size_t i = 0; i < m; ++i) {
            hist_min_ = (x[i] != 0.0) ? std::min(hist_min_, x_[i]) : hist_min_;
            hist_max_ = std::max(hist_max_, x_[i]);
        }
        hist_max_ *= (1.0 + 1e-6);
    }

    BMCI(const BMCI &)  = default;
    BMCI(      BMCI &&) = default;
    BMCI & operator=(const BMCI &)  = default;
    BMCI & operator=(      BMCI &&) = default;

    ~BMCI()
    {
        // Memory is managed by std::unique_ptr.
    };

    template<typename T>
    void expectation(TFloat *x_hat,  const T *Y, size_t m);
    template<typename T>
    void pdf(TFloat *x_hat, TFloat *hist, const T *Y, size_t m, size_t n_bins);

private:

    #pragma omp declare simd
    TFloat scaled_dot(TFloat y1, TFloat y2, TFloat s)
    {
        TFloat dy = y1 - y2;
        return dy * dy * s;
    }

    inline size_t get_bin(TFloat x, TFloat hist_min, TFloat d_hist);
    size_t m_, n_;

    PaddedMatrix<TFloat, 1, 1, alignment> Y_;
    PaddedVector<TFloat, 1, alignment> s_inv_;
    PaddedVector<TFloat, 1, alignment> x_;

    TFloat hist_min_, hist_max_;
};

// Static Members
// template <typename TFloat, template<typename> class TMem>
// size_t BMCI<TFloat, TMem>::alignment = TMem<TFloat>::alignment;

// template <typename TFloat, template<typename> class TMem>
// size_t BMCI<TFloat, TMem>::array_padding = TMem<TFloat>::array_padding;

// template <typename TFloat, template<typename> class TMem>
// size_t BMCI<TFloat, TMem>::s_padding = TMem<TFloat>::padding / sizeof(TFloat);

// Member Functions
template <typename TFloat, template<typename> class TMem>
size_t BMCI<TFloat, TMem>::get_bin(TFloat x, TFloat hist_min, TFloat d_hist)
{
    if (x == 0.0) {
        return 0;
    } else {
        TFloat i_f = (x - hist_min) / d_hist;
        return 1 + std::trunc(i_f);
    }
}

template <typename TFloat, template<typename> class TMem>
template<typename T>
void BMCI<TFloat, TMem>::expectation(TFloat *x_hat, const T *Y, size_t m)
{
    constexpr size_t meas_block_size = 4;
    constexpr size_t sim_block_size = 4;

    size_t n_meas_blocks = m_ / meas_block_size;
    size_t n_sim_blocks  = m  / sim_block_size;
    size_t rem_meas_blocks = m_ % meas_block_size;
    size_t rem_sim_blocks  = m  % sim_block_size;

    PaddedMatrix<TFloat, 1, 12, alignment> Y_t(m, n_);
    PaddedMatrix<TFloat, 1, 1, alignment>   p(meas_block_size, m);
    PaddedVector<TFloat, 1, alignment>     p_sum(m);
    PaddedVector<TFloat, 1, alignment>     px(m);

    Y_t.copy(Y);

    // Initialize x_hat and p_sum
    for (size_t i = 0; i < m; i++) {
        x_hat[i] = 0.0;
        p_sum[i] = 0.0;
    }

    for (size_t ind_sim_blocks = 0; ind_sim_blocks < n_sim_blocks; ++ind_sim_blocks) {
        // Compute arguments to exp.
        for (size_t ind_meas_blocks = 0; ind_meas_blocks < n_meas_blocks; ++ind_meas_blocks) {
            
        }
    }

    
    // Outer loop over database entries.
    size_t sim_ind(0);
    for (size_t i = 0; i < m_; ++i) {
        // Inner loop over measurements.
        size_t meas_ind(0);
        for (size_t j = 0; j < m; ++j) {
            TFloat p(0.0), dy(0.0), dySdy(0.0);
            #pragma omp simd safelen(row_padding)
            for (size_t k = 0; k < n_; ++k) {
                dy = Y_(i, k) - Y_t(j, k);
                dySdy += dy * dy * s_inv_[k];
            }
            p = exp(-dySdy);
            p_sum[j] += p;
            x_hat[j] += p * x_[i];
            meas_ind += row_padding;
        }
        sim_ind += row_padding;
    }

    // Normalize by p_sum.
    for (size_t i = 0; i < m; i++) {
        x_hat[i] /= p_sum[i];
    }
}

template <typename TFloat, template<typename> class TMem>
template <typename T>
void BMCI<TFloat, TMem>::pdf(TFloat *x_hat, TFloat *hist, const T *Y, size_t m, size_t n_bins)
{
    TFloat *dY = new TFloat[n_];
    TFloat hist_min_log = log(hist_min_);
    TFloat d_hist = (log(hist_max_) - hist_min_log) / (n_bins - 1);
    size_t Y_i_ind(0), hist_ind(0);

    PaddedMatrix<TFloat, 1, 12, alignment> Y_t(m, n_);

    for (size_t i = 0; i < m; i++) {
        x_hat[i] = 0.0;
        TFloat p_sum(0.0);
        size_t Y_j_ind(0);
        for (size_t j = 0; j < m_; j++) {
            TFloat p(0.0), dy(0.0), dySdy(0.0);
            for (size_t k = 0; k < n_; k++) {
                dy = Y_(j, k) - Y_t(i, k);
                dySdy += dy * dy * s_inv_[k];
            }
            p = exp(-dySdy);
            p_sum += p;
            x_hat[i] += p * x_[j];
            Y_j_ind += row_padding;

            // Add p to histogram bin.
            TFloat x_log = (x_[j] == 0.0) ? 0.0 : log(x_[j]);
            hist[hist_ind + get_bin(x_log, hist_min_log, d_hist)] += p;

        }
        x_hat[i] /= p_sum;
        Y_i_ind += row_padding;
        hist_ind += n_bins;
    }
}

////////////////////////////////////////////////////////////////////////////////
// The Python Interface
////////////////////////////////////////////////////////////////////////////////
#include <Python.h>
#include <numpy/npy_common.h>
#include <numpy/arrayobject.h>

extern "C" {

    BMCI<float>  *bmci_float;
    BMCI<double> *bmci_double;

    static PyObject *
    vec_exp(PyObject *self, PyObject *args)
    {
        PyObject *x_array;

        if (!PyArg_ParseTuple(args, "OOO", &x_array)) {
            return NULL;
        }

        if (PyArray_TYPE(x_array) == NPY_FLOAT32) {
            exponential(static_cast<float*>(PyArray_DATA(x_array)));
        } else {
            exponential(static_cast<double*>(PyArray_DATA(x_array)));
        }
        return x;
    }

    static PyObject *
    vec_scaled_dot(PyObject *self, PyObject *args)
    {
        PyObject *y_1_array, *y_2_array, *s_array;

        if (!PyArg_ParseTuple(args, "OOO", &y_1_array, &y_2_array, &s_array)) {
            return NULL;
        }

        std::cout << "shoot" << std::endl;
        PyObject *x;
        if (PyArray_TYPE(y_1_array) == NPY_FLOAT32) {
            npy_intp dim = 1;
            x = PyArray_SimpleNew(1, &dim, NPY_FLOAT32);
            *static_cast<float*>(PyArray_DATA(x)) = scaled_dot_product<float, 16>(static_cast<float*>(PyArray_DATA(y_1_array)),
                                                                                  static_cast<float*>(PyArray_DATA(y_2_array)),
                                                                                  static_cast<float*>(PyArray_DATA(s_array)));
        } else {
            npy_intp dim = 1;
            x = PyArray_SimpleNew(1, &dim, NPY_FLOAT64);
            *static_cast<double*>(PyArray_DATA(x)) = scaled_dot_product<double, 16>(static_cast<double*>(PyArray_DATA(y_1_array)),
                                                                                   static_cast<double*>(PyArray_DATA(y_2_array)),
                                                                                   static_cast<double*>(PyArray_DATA(s_array)));
        }
        std::cout << "dead" << std::endl;
        return x;
    }

    static PyObject *
    bmci_initialize(PyObject *self, PyObject *args)
    {
        // Parse Arguments.
        PyObject *Y_array, *x_array, *s_array;
        Py_ssize_t prec;

        if (!PyArg_ParseTuple(args, "OOOn", &Y_array, &x_array, &s_array, &prec)) {
            return NULL;
        }
        size_t m(0), n(0);
        m = (size_t) PyArray_DIMS(Y_array)[0];
        n = (size_t) PyArray_DIMS(Y_array)[1];

        if (prec == 32) {
            bmci_float = static_cast<BMCI<float>*>(
                new BMCI<float>(static_cast<double*>(PyArray_DATA(Y_array)),
                                static_cast<double*>(PyArray_DATA(x_array)),
                                static_cast<double*>(PyArray_DATA(s_array)),
                                m, n));
        } else if (prec == 64) {
            bmci_double = static_cast<BMCI<double>*>(
                new BMCI<double>(static_cast<double*>(PyArray_DATA(Y_array)),
                                 static_cast<double*>(PyArray_DATA(x_array)),
                                 static_cast<double*>(PyArray_DATA(s_array)),
                                 m, n));
        } else {
            return NULL;
        }

        Py_RETURN_NONE;
    }

    static PyObject *
    bmci_finalize(PyObject */*self*/, PyObject */*args*/)
    {
        if (bmci_float) {
            delete bmci_float;
        }
        if (bmci_double) {
            delete bmci_double;
        }
        Py_RETURN_NONE;
    }

    static PyObject *
    bmci_expectation(PyObject *self, PyObject *args)
    {
        // Parse Arguments.
        PyObject *Y_array;
        if (!PyArg_ParseTuple(args, "O", &Y_array)) {
            return NULL;
        }
        size_t m(0);
        m    = (size_t) PyArray_DIMS(Y_array)[0];

        npy_intp *dims = new npy_intp[2];
        dims[0] = m;
        dims[1] = 1;

        // Run BMCI depending on type.
        PyObject *x;
        if ((PyArray_TYPE(Y_array) == NPY_FLOAT32) && bmci_float) {
            x = PyArray_SimpleNew(2, dims, NPY_FLOAT32);
            bmci_float->expectation(static_cast<float*>(PyArray_DATA(x)),
                                    static_cast<float*>(PyArray_DATA(Y_array)),
                                    m);
        } else if ((PyArray_TYPE(Y_array) == NPY_FLOAT64) && bmci_double) {
            x = PyArray_SimpleNew(2, dims, NPY_FLOAT64);
            bmci_double->expectation(static_cast<double*>(PyArray_DATA(x)),
                                     static_cast<double*>(PyArray_DATA(Y_array)),
                                     m);
        } else {
            return NULL;
        }

        // Clean up.
        delete[] dims;
        return x;
    }

    static PyObject *
    bmci_pdf(PyObject *self, PyObject *args)
    {
        // Parse Arguments.
        PyObject *Y_array;
        Py_ssize_t n_bins;
        if (!PyArg_ParseTuple(args, "On", &Y_array, &n_bins)) {
            return NULL;
        }
        size_t m(0);
        m    = (size_t) PyArray_DIMS(Y_array)[0];

        // Run BMCI depending on type.
        npy_intp *x_dims = new npy_intp[2];
        x_dims[0] = m;
        x_dims[1] = 1;
        npy_intp *hist_dims = new npy_intp[2];
        hist_dims[0] = m;
        hist_dims[1] = n_bins;

        PyObject *x, *hist;
        if ((PyArray_TYPE(Y_array) == NPY_FLOAT32) && bmci_float) {
            x = PyArray_SimpleNew(2, x_dims, NPY_FLOAT32);
            hist = PyArray_ZEROS(2, hist_dims, NPY_FLOAT32, 0);
            bmci_float->pdf(static_cast<float*>(PyArray_DATA(x)),
                            static_cast<float*>(PyArray_DATA(hist)),
                            static_cast<float*>(PyArray_DATA(Y_array)),
                            m, n_bins);
        } else if ((PyArray_TYPE(Y_array) == NPY_FLOAT64) && bmci_double) {
            x = PyArray_SimpleNew(2, x_dims, NPY_FLOAT64);
            hist = PyArray_ZEROS(2, hist_dims, NPY_FLOAT64, 0);
            bmci_double->pdf(static_cast<double*>(PyArray_DATA(x)),
                             static_cast<double*>(PyArray_DATA(hist)),
                             static_cast<double*>(PyArray_DATA(Y_array)),
                             m, n_bins);
        } else {
            return NULL;
        }

        // Zip arrays.
        PyObject *pair = PyTuple_New(2);
        PyTuple_SetItem(pair, 0, x);
        PyTuple_SetItem(pair, 1, hist);

        // Clean up.
        delete[] x_dims;
        delete[] hist_dims;

        // Done.
        return pair;
    }

    static PyMethodDef methods[] = {
        {"vec_exp",  vec_exp, METH_VARARGS, "Vectorized implementation of the exponential function."},
        {"vec_scaled_dot",  vec_scaled_dot, METH_VARARGS, "Vectorized implementation of scaled dot product."},
        {"bmci_initialize",  bmci_initialize, METH_VARARGS, "Initialize bmci object."},
        {"bmci_finalize",  bmci_finalize, METH_NOARGS, "Finalize bmci object."},
        {"bmci_expectation",  bmci_expectation, METH_VARARGS, "Compute expectation value of the retrieval using BMCI"},
        {"bmci_pdf",  bmci_pdf, METH_VARARGS, "Compute expectation and PDF of the posterior value of the retrieval using BMCI"},
        {NULL, NULL, 0, NULL}        /* Sentinel */
    };

    static struct PyModuleDef bmci_module = {
        PyModuleDef_HEAD_INIT,
        "bmci",   /* name of module */
        NULL,
        -1,
        methods
    };

    PyMODINIT_FUNC
    PyInit_bmcixx(void)
    {
        import_array();
        return PyModule_Create(&bmci_module);
    }
}
