#include <stdlib.h>
#include <cmath>
#include <iostream>
#include <limits>
#include <algorithm>
#include <memory>

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
        : m_(m), n_(n), hist_min_(std::numeric_limits<TFloat>::max()), hist_max_(0.0)
    {
        // Aligned Memory
        std::tie(Y_ptr_, Y_)         = allocate_matrix_aligned(m, n, alignment, row_padding);
        std::tie(x_ptr_, x_)         = allocate_vector_aligned(m, alignment);
        std::tie(s_inv_ptr_, s_inv_) = allocate_matrix_aligned(1, n, alignment, row_padding);

        // Copy Data
        copy_matrix_padded(Y_, Y, m);
        copy_matrix_padded(s_inv_, s, 1);
        copy_vector(x_, x, m);

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

    std::pair<std::unique_ptr<TFloat[]>, TFloat*> allocate_matrix_aligned(size_t m, size_t n, size_t alignment, size_t row_padding)
    {
        size_t n_elements = m * row_padding;
        size_t n_padding  = std::max<size_t>(alignment / sizeof(TFloat), 1);
        size_t size       = n_elements * sizeof(TFloat);
        size_t space      = (n_elements + n_padding) * sizeof(TFloat);
        std::unique_ptr<TFloat[]> mat_ptr = std::make_unique<TFloat[]>(n_elements + n_padding);

        void *ptr = reinterpret_cast<void*>(mat_ptr.get());
        if (std::align(alignment, size, ptr, space)) {
            return std::make_pair(std::move(mat_ptr), reinterpret_cast<TFloat*>(ptr));
        } else {
            throw std::runtime_error("Couldn't allocate matrix.");
        }
        return std::make_pair(nullptr, nullptr);
    }

    std::pair<std::unique_ptr<TFloat[]>, TFloat*> allocate_vector_aligned(size_t m, size_t alignment)
        {
            size_t n_elements = m;
            size_t n_padding  = std::max<size_t>(alignment / sizeof(TFloat), 1);
            size_t size       = n_elements * sizeof(TFloat);
            size_t space      = (n_elements + n_padding) * sizeof(TFloat);
            std::unique_ptr<TFloat[]> vec_ptr = std::make_unique<TFloat[]>(n_elements + n_padding);

            void *ptr = reinterpret_cast<void*>(vec_ptr.get());
            if (std::align(alignment, size, ptr, space)) {
                return std::make_pair(std::move(vec_ptr), reinterpret_cast<TFloat*>(ptr));
            } else {
                throw std::runtime_error("Couldn't allocate matrix.");
            }
            return std::make_pair(nullptr, nullptr);
        }

    template <typename T>
    void copy_matrix_padded(TFloat * dest, const T *src, size_t m) {
        for (size_t i = 0; i < m; i++) {
            for (size_t j = 0; j < row_padding; j++) {
                if (j < n_) {
                    dest[i * row_padding + j] = src[i * n_ + j];
                } else {
                    dest[i * row_padding + j] = 0.0;
                }
            }
        }
    }

    template <typename T>
    void copy_vector(TFloat * dest, const T *src, size_t m) {
        for (size_t i = 0; i < m; i++) {
            dest[i] = src[i];
        }
    }

    #pragma omp declare simd
    TFloat scaled_dot(TFloat y1, TFloat y2, TFloat s)
    {
        TFloat dy = y1 - y2;
        return dy * dy * s;
    }

    inline size_t get_bin(TFloat x, TFloat hist_min, TFloat d_hist);
    size_t m_, n_;

    // Unique_ptr for memory management.
    std::unique_ptr<TFloat[]> Y_ptr_, x_ptr_, s_inv_ptr_;
    // Aligned arrays.
    TFloat *Y_, *x_, *s_inv_;

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
    TFloat *dY    = new TFloat[n_];
    TFloat *p_sum = new TFloat[m];
    std::unique_ptr<TFloat[]> Y_t_ptr(nullptr);
    TFloat *Y_t(nullptr);

    // Copy input data to aligned array.
    std::tie(Y_t_ptr, Y_t) = allocate_matrix_aligned(m, n_, alignment, row_padding);
    copy_matrix_padded(Y_t, Y, m);

    // Initialize x_hat and p_sum
    for (size_t i = 0; i < m; i++) {
        x_hat[i] = 0.0;
        p_sum[i] = 0.0;
    }


    // Outer loop over database entries.
    size_t sim_ind(0);
    for (size_t i = 0; i < m_; ++i) {
        // Inner loop over measurements.
        size_t meas_ind(0);
        for (size_t j = 0; j < m; ++j) {
            TFloat *Y_sim  = Y_ + sim_ind;
            TFloat *Y_meas = Y_t + meas_ind;
            TFloat p(0.0), dy(0.0), dySdy(0.0);
            #pragma omp simd safelen(row_padding) aligned(Y_sim, Y_meas)
            for (size_t k = 0; k < row_padding; ++k) {
                TFloat t = scaled_dot(Y_sim[k], Y_meas[k], s_inv_[k]);
                dySdy += t;
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
    std::unique_ptr<TFloat[]> Y_t_ptr(nullptr);
    TFloat *Y_t(nullptr);

    // Copy input data to aligned array.
    std::tie(Y_t_ptr, Y_t) = allocate_matrix_aligned(m, n_, alignment, row_padding);
    copy_matrix_padded(Y_t, Y, m);

    for (size_t i = 0; i < m; i++) {
        x_hat[i] = 0.0;
        TFloat p_sum(0.0);
        size_t Y_j_ind(0);
        for (size_t j = 0; j < m_; j++) {
            TFloat p(0.0), dy(0.0), dySdy(0.0);
            for (size_t k = 0; k < n_; k++) {
                dy = Y_[Y_j_ind + k] - Y_t[Y_i_ind + k];
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
