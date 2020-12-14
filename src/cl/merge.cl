
__kernel void mergeSort(__global const float* xs, __global float* res,
                         const unsigned int n, const unsigned int m){
    const unsigned gx = get_global_id(0);
    
    if(gx >= n){
        return;
    }

    int shiftLeft = (gx/(m << 1)) * (m  << 1);
    int shiftRight = shiftLeft + m;
    //min(shiftRight + m, n);
    if(shiftRight > n){
        shiftRight = n;
    }

    int lastRight = shiftRight + m;
    if(lastRight > n){
        lastRight = n;
    }

    int lx = gx - shiftLeft;// 

    int left = lx + shiftRight - lastRight;
    if(left < 0){
        left = 0;
    }

    int right = lx;
    if(lx > m){
        right = m;
    }



    while(left < right){
        int middle = (left + right) >> 1;

        int l = shiftLeft + middle;
        int r = shiftRight + lx - middle - 1;

        if(xs[l] < xs[r]){
            left = middle + 1;
        }
        else{
            right = middle;
        }
    }

    int l = shiftLeft + left;
    int r = shiftRight + lx - left;

    if((l < shiftRight) && (xs[l] < xs[r] || (lastRight <= r))){
        res[gx] = xs[l];
    }
    else{
        res[gx] = xs[r];
    }

}
