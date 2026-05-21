from outlier_migrate.quantization import quantize_matrix_per_output_channel, symmetric_int4_quantize_row


def test_symmetric_int4_quantize_row_preserves_zero_row() -> None:
    assert symmetric_int4_quantize_row([0.0, 0.0]) == [0.0, 0.0]


def test_quantize_matrix_restores_protected_row_and_column() -> None:
    matrix = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ]
    quantized = quantize_matrix_per_output_channel(matrix, protected_rows={1}, protected_cols={0})
    assert quantized[1] == matrix[1]
    assert quantized[0][0] == matrix[0][0]
    assert len(quantized) == 2
