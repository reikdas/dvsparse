struct COO {
    int rows, cols, nnz;
    std::vector<int> row_indices;
    std::vector<int> col_indices;
    std::vector<double> values;

    void remove_duplicates() {
        // write a function to remove duplicates using set
        std::set<std::tuple<int, int, double>> unique_entries;
        for (int i = 0; i < nnz; i++) {
            if (col_indices[i] >= cols || row_indices[i] >= rows) {
                std::cerr << "Error: Invalid matrix entry\n";
                exit(1);
            }
            unique_entries.insert({row_indices[i], col_indices[i], values[i]});
        }

        row_indices.clear();
        col_indices.clear();
        values.clear();

        for (const auto &entry : unique_entries) {
            row_indices.push_back(std::get<0>(entry));
            col_indices.push_back(std::get<1>(entry));
            values.push_back(std::get<2>(entry));
        }
        nnz = row_indices.size();
    }

    // Sort by (row, column)
    void sort() {
        std::vector<std::tuple<int, int, double>> entries;
        for (int i = 0; i < nnz; i++) {
            entries.push_back({row_indices[i], col_indices[i], values[i]});
        }

        std::sort(entries.begin(), entries.end(), [&](const auto &a, const auto &b) {
            return std::tie(std::get<0>(a), std::get<1>(a)) < std::tie(std::get<0>(b), std::get<1>(b));
        });

        for (int i = 0; i < nnz; i++) {
            row_indices[i] = std::get<0>(entries[i]);
            col_indices[i] = std::get<1>(entries[i]);
            values[i] = std::get<2>(entries[i]);
        }
    }
};

void iterate_and_check(const COO &coo) {
    for (int i = 0; i < coo.nnz; i++) {
        if (coo.row_indices[i] >= coo.rows || coo.col_indices[i] >= coo.cols) {
            std::cerr << "Error: Invalid matrix entry3\n";
            exit(1);
        }
    }
}

// Define CSR struct without using vectors
struct CSR {
    int rows, cols, nnz;
    int *row_ptrs;
    int *col_indices;
    double *values;
};

// add a function to convert COO to CSR
CSR coo_to_csr(const COO &coo) {
    CSR csr;
    csr.rows = coo.rows;
    csr.cols = coo.cols;
    csr.nnz = coo.nnz;

    csr.row_ptrs = new int[csr.rows + 1];
    csr.col_indices = new int[csr.nnz];
    csr.values = new double[csr.nnz];

    std::vector<int> csr_row_ptr;

    int nnz = coo.nnz;
    csr_row_ptr.assign(csr.rows + 1, 0);

        // Compute row_ptr (row starts)
    for (int i = 0; i < nnz; i++) {
        csr_row_ptr[coo.row_indices[i] + 1]++;
        csr.col_indices[i] = coo.col_indices[i];
        csr.values[i] = coo.values[i];
    }

    // Compute prefix sum to determine row_ptr positions
    for (int i = 0; i < csr.rows; i++) {
        csr_row_ptr[i + 1] += csr_row_ptr[i];
    }

    // Copy the row_ptr to the CSR matrix
    for (int i = 0; i < csr.rows + 1; i++) {
        csr.row_ptrs[i] = csr_row_ptr[i];
    }

    return csr;
}

// add enum to keep track of the matrix data type
enum class MatrixType {
    REAL,
    INTEGER,
    PATTERN
};

// add enum to keep track of the matrix format general, symmetric
enum class MatrixFormat {
    GENERAL,
    SYMMETRIC
};

COO readMTXtoCOO(const std::string &filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        exit(1);
    }

    MatrixFormat matrixFormat;
    MatrixType matrixType;

    std::string line;
    int i = 0;
    while (std::getline(file, line)) {
        // line 0 should contain MatrixMarket header, the matrix type and the matrix format
        if (i == 0) {
            if (line.find("MatrixMarket") != std::string::npos) {
                if (line.find("real") != std::string::npos) {
                    matrixType = MatrixType::REAL;
                } else if (line.find("integer") != std::string::npos) {
                    matrixType = MatrixType::INTEGER;
                } else if (line.find("pattern") != std::string::npos) {
                    matrixType = MatrixType::PATTERN;
                } else {
                    std::cerr << "Error: Invalid MatrixMarket header\n";
                    exit(1);
                }

                if (line.find("general") != std::string::npos) {
                    matrixFormat = MatrixFormat::GENERAL;
                } else if (line.find("symmetric") != std::string::npos) {
                    matrixFormat = MatrixFormat::SYMMETRIC;
                }
                i++;
            } else {
                std::cerr << "Error: Invalid MatrixMarket header\n";
                exit(1);
            }
            continue;
        }
        // Skip comments (lines starting with '%')
        else if (line[0] == '%') {
            i++;
            continue;
        }
        break; // First non-comment line contains matrix size
    }

    std::istringstream iss(line);
    int M, N, nz;
    if (!(iss >> M >> N >> nz)) {
        std::cerr << "Error: Invalid matrix size format\n";
        exit(1);
    }

    if (M != N && matrixFormat == MatrixFormat::SYMMETRIC) {
        std::cerr << "Error: Symmetric matrix must be square\n";
        exit(1);
    }


    COO coo;
    coo.rows = M;
    coo.cols = N;

    if (matrixFormat == MatrixFormat::SYMMETRIC) {
        coo.row_indices.reserve(nz*2);
        coo.col_indices.reserve(nz*2);
        coo.values.reserve(nz*2);
        coo.nnz = nz*2;
    } else {
        coo.row_indices.reserve(nz);
        coo.col_indices.reserve(nz);
        coo.values.reserve(nz);
        coo.nnz = nz;
    }

    // Read matrix entries
    for (int i = 0; i < nz; i++) {
        int row, col;
        double value;

        if (matrixType == MatrixType::REAL) {
            double double_value;
            if (!(file >> row >> col >> double_value)) {
                std::cerr << "Error: Invalid matrix entry format\n";
                exit(1);
            }
            value = (double) double_value;
        } else if (matrixType == MatrixType::INTEGER) {
            int int_value;
            if (!(file >> row >> col >> int_value)) {
                std::cerr << "Error: Invalid matrix entry format\n";
                exit(1);
            }
            value = (double) int_value;
        } else if (matrixType == MatrixType::PATTERN) {
            if (!(file >> row >> col)) {
                std::cerr << "Error: Invalid matrix entry format\n";
                exit(1);
            }
            value = 1.0;
        }

        coo.row_indices.push_back(row - 1); // Convert 1-based to 0-based
        coo.col_indices.push_back(col - 1);
        coo.values.push_back(value);

        if (matrixFormat == MatrixFormat::SYMMETRIC && row != col) {
            coo.row_indices.push_back(col - 1);
            coo.col_indices.push_back(row - 1);
            coo.values.push_back(value);
        }
    }

    // sort the COO matrix by row and col indices
    coo.remove_duplicates(); // commenting out for now
    coo.sort();

    file.close();
    return coo;
}