__kernel void aplusb(__global const float * as,
                     __global const float * bs,
                     __global float * cs,
                     unsigned int inputsDim)
{

    size_t workItemId = get_global_id(0);
    if (inputsDim < workItemId) {
        return;
    }

    cs[workItemId] = as[workItemId] + bs[workItemId];
}