#define WORK_GROUP_SIZE 256

__kernel void bitonic_local(__global float* as, unsigned int batch_size, unsigned int size, unsigned int n) {
	unsigned int local_id = get_local_id(0);
	unsigned int global_id = get_global_id(0);

	__local float batch[WORK_GROUP_SIZE];

	if (global_id < n) {
		batch[local_id] = as[global_id];
	}

	barrier(CLK_LOCAL_MEM_FENCE);
	
	// ����� ��������� "���" ������������������
	// 1 - ������������
	// 0 - ���������
	int tone_flag = global_id % (2 * size) < size;

	while (batch_size >= 1) {

		if (global_id % (2 * batch_size) < batch_size && global_id + batch_size < n) {
			float a = batch[local_id];
			float b = batch[local_id + batch_size];
			// � ������, ���� ������� ��������� �������
			// �������� (a > b) ����� ����� tone_flag
			if ((a > b) == tone_flag) {
				batch[local_id] = b;
				batch[local_id + batch_size] = a;
			}
		}

		barrier(CLK_LOCAL_MEM_FENCE);
		batch_size >>= 1;
	}

	if (global_id < n) {
		as[global_id] = batch[local_id];
	}
}


__kernel void bitonic(__global float* as, unsigned int batch_size, unsigned int size, unsigned int n) {
	unsigned int global_id = get_global_id(0);

	// ����� ��������� "���" ������������������
	// 1 - ������������
	// 0 - ���������
	int tone_flag = global_id % (2 * size) < size;

	if (global_id % (2 * batch_size) < batch_size && global_id + batch_size < n) {
		float a = as[global_id];
		float b = as[global_id + batch_size];
		// � ������, ���� ������� ��������� �������
		// �������� (a > b) ����� ����� tone_flag
		if ((a > b) == tone_flag) {
			as[global_id] = b;
			as[global_id + batch_size] = a;
		}
	}
}
