__kernel void merge(__global const float* a, __global float* res, int n, int k) {
	int i = get_global_id(0);
	int shift = i - (i % (2 * k));
	
	//a[i] --> a[i + shift]
	//b[i] --> a[i + shift + k]
	// kXk --> 2k after merge
	
	int d = i % (2 * k);
	if (d == 2 * k - 1) {
		res[i] = max(a[k - 1 + shift], a[k - 1 + shift + k]);
	} else {
		int l = 0;
		int r = min(d, k) + 1;
		while (r - l > 1) {
			int m = (l + r) / 2;
			//if (i < 16 && i >= 12)
			//	printf("i = %d, d = %d, l = %d, m = %d, r = %d, i1 = %d, i2 = %d\n", i, d, l, m, r, m - 1, d - m);
			if (d - m >= k || a[m - 1 + shift] <= a[d - m + shift + k]) {
				l = m;
			} else {
				r = m;
			}
		}
		//if (i < 16 && i >= 12)
		//	printf("i = %d, d = %d, l = %d, k = %d, l + shift = %d, d - l + shift + k = %d\n", i, d, l, k, l + shift, d - l + shift + k);
		if (d - l < k && (l >= k || a[l + shift] >= a[d - l + shift + k])) {
			res[i] = a[d - l + shift + k];
		} else {
			res[i] = a[l + shift];
		}
	}
}
