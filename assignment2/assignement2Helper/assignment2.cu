// Includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

// Includes CUDA
#include <cuda_runtime.h>

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check

#define filter_width 15
#define filter_height 15
#define TILE_WIDTH 32 + 2*(filter_width/2)

// Define the files that are to be save and the reference images for validation

const char *imageFilename = "image21.pgm";

const char *sampleName = "simpleTexture";
int option = 4;


////////////////////////////////////////////////////////////////////////////////
// Constants
// Texture reference for 2D float texture
texture<float, 2, cudaReadModeElementType> tex;

//declare constant memory
__constant__ float d_filter_constant[filter_width*filter_height];

// Auto-Verification Code
bool testResult = true;

////////////////////////////////////////////////////////////////////////////////
//! Convolute image
//! @param outputData  output data in global memory
////////////////////////////////////////////////////////////////////////////////

void cpuConvolution(float *image,
                    float *outImage,
                    float *filter,
                    int width,
                    int height,
                    int pad_w,
                    int pad_h)
{
    //time vars
    clock_t start, end;
    start = clock();
    //calculate the convolution per image pixel
    for(int i = 0; i < height; i++){
        for(int j = 0; j < width; j++){
            double convolution_value = 0;
            for(int k = 0; k < filter_height; k++){
                for(int l = 0; l < filter_width; l++){
                    convolution_value += filter[l+k*filter_width]*image[((j+l)+(i+k)*(width+(2*pad_w)))];
                }
            }
            // write pixel to output array
            outImage[j + i*width] = convolution_value;
        }
    }
    end = clock();
    printf("Processing time: %f (ms)\n", (double)((end-start)/1000));
}

__global__ void naiveConvolutionKernel(float *image,
                                      float *outImage,
                                      float *filter,
                                      int width,
                                      int pad_w)
{
    // calculate coordinates
    int i = threadIdx.y + blockIdx.y*blockDim.y;
    int j = threadIdx.x + blockIdx.x*blockDim.x;

    //calculate the convolution per image pixel
    double convolution_value = 0;
    for(int k = 0; k < filter_height; k++){
        for(int l = 0; l < filter_width; l++){
            convolution_value += filter[l+k*filter_width]*image[((j+l)+(i+k)*(width+(2*pad_w)))];
        }
    }

    // save value to output data
    outImage[j + i*width] = convolution_value;
}

__global__ void sharedConvolutionKernel(float *image,
                                      float *outImage,
                                      float *filter,
                                      int width,
                                      int pad_w)
{
    //allocate shared memory space
    __shared__ float s_image[TILE_WIDTH][TILE_WIDTH];
    __shared__ float s_filter[filter_height][filter_width];

    // calculate the row & column index of the element
    int blockx = blockIdx.x;
    int blocky = blockIdx.y;
    int threadx = threadIdx.x;
    int thready = threadIdx.y;
    int i = blocky*(TILE_WIDTH - (2*pad_w)) + thready;
    int j = blockx*(TILE_WIDTH - (2*pad_w))  + threadx;

    // load data into shared memory

    // load filter into shared memory in 1 thread per block along with the image tile we wish to use for the convolution of this block

    for(int k = 0; k < filter_height; k++){
        for(int l = 0; l < filter_width; l++){
			if(threadx == 0 && thready == 0){
				s_filter[k][l] = filter[l+k*filter_width];
			}
			s_image[thready+k][threadx+l] = image[(j+l)+((i+k)*(width+(2*pad_w)))];
        }
    }
    __syncthreads();


    //calculate the convolution per image pixel using shared memory
    double convolution_value = 0;
    for(int k = 0; k < filter_height; k++){
        for(int l = 0; l < filter_width; l++){
            convolution_value += s_filter[k][l]*s_image[thready+k][threadx+l];
        }
    }

    // save value to output data
    outImage[j + i*width] = convolution_value;
}

__global__ void constantSharedConvolutionKernel(float *image,
                                      float *outImage,
                                      int width,
                                      int pad_w)
{
    //allocate shared memory space
    __shared__ float s_image[TILE_WIDTH][TILE_WIDTH];

    // calculate the row & column index of the element
    int blockx = blockIdx.x;
    int blocky = blockIdx.y;
    int threadx = threadIdx.x;
    int thready = threadIdx.y;
    int i = blocky*(TILE_WIDTH - (2*pad_w)) + thready;
    int j = blockx*(TILE_WIDTH - (2*pad_w))  + threadx;

    // load data into shared memory

    //  load the image tile we wish to use for the convolution of this block into shared memory

    for(int k = 0; k < filter_height; k++){
        for(int l = 0; l < filter_width; l++){
			s_image[thready+k][threadx+l] = image[(j+l)+((i+k)*(width+(2*pad_w)))];
        }
    }
    __syncthreads();

    //calculate the convolution per image pixel using shared memory
    double convolution_value = 0;
    for(int k = 0; k < filter_height; k++){
        for(int l = 0; l < filter_width; l++){
            convolution_value += d_filter_constant[l+k*filter_width]*s_image[thready+k][threadx+l];
        }
    }

    // save value to output data
    outImage[j + i*width] = convolution_value;
}

__global__ void textureConvolutionKernel(float *outImage,
                                      int width,
                                      int height,
                                      int img_width)
{
    // calculate coordinates
    int i = threadIdx.y + blockIdx.y*blockDim.y;
    int j = threadIdx.x + blockIdx.x*blockDim.x;

    //calculate the convolution per image pixel
    double convolution_value = 0;
    for(int k = 0; k < filter_height; k++){
        for(int l = 0; l < filter_width; l++){
            convolution_value += d_filter_constant[l+k*filter_width]*tex2D(tex, j - height + k + 0.5f, i - width + l + 0.5f);
        }
    }

    // save value to output data
    outImage[j + (i*(img_width))] = convolution_value;
}
////////////////////////////////////////////////////////////////////////////////
// Declaration, forward
void runTest(int argc, char **argv);

void filter_gen(float* filter, int width, int height);

void padded_populate(float* padded, float* image, int width, int height/*, float *filter*/);

void executeNaiveConvolution(int height, int width, float *d_padded_image, float *dData, float *d_filter, int padWidth);

void executeSharedConvolution(int height, int width, float *d_padded_image, float *dData, float *d_filter, int padWidth);

void executeConstantSharedConvolution(int height, int width, float *d_padded_image, float *dData, int padWidth);

void executeTextureConvolution(int height, int width, float *dData, int padHeight, int padWidth);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    //printf("%s starting...\n", sampleName);

    // Process command-line arguments
    if (argc > 1)
    {
        if (checkCmdLineFlag(argc, (const char **) argv, "input"))
        {
            getCmdLineArgumentString(argc,
                                     (const char **) argv,
                                     "input",
                                     (char **) &imageFilename);
        }
    }
    for(int i = 0; i<=4; i++){
        option = i;
        runTest(argc, argv);
    }


    cudaDeviceReset();
    //printf("%s completed, returned %s\n",
    //       sampleName,
    //       testResult ? "OK" : "ERROR!");
    exit(testResult ? EXIT_SUCCESS : EXIT_FAILURE);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void runTest(int argc, char **argv)
{
    //method name
    char * name;
    //int devID = findCudaDevice(argc, (const char **) argv);
    // load image from disk
    float *hData = NULL;
    //host filter
    float* h_filter = NULL;
    //width and height of original image
    unsigned int width, height;
    char *imagePath = sdkFindFilePath(imageFilename, argv[0]);

    if (imagePath == NULL)
    {
        printf("Unable to source image file: %s\n", imageFilename);
        exit(EXIT_FAILURE);
    }
    // load image into hData
    sdkLoadPGM(imagePath, &hData, &width, &height);

    ////////////////////////////////////
    // generate filter
    ////////////////////////////////////

    h_filter = (float *)malloc(filter_width*filter_height*sizeof(float));
    filter_gen(h_filter,filter_width,filter_height);

    //total size of image
    unsigned int size = width * height * sizeof(float);
    // Allocate mem for the result on host side
    float *hOutputData = (float *) malloc(size);

    //printf("Loaded '%s', %d x %d pixels\n", imageFilename, width, height);

    //option 0 is texture
    if(option == 0){

        // array for output image
        float *dData = NULL;
        checkCudaErrors(cudaMalloc((void **) &dData, size));

        // Allocate array and copy image data
        cudaChannelFormatDesc channelDesc =
            cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
        cudaArray *cuArray;
        checkCudaErrors(cudaMallocArray(&cuArray,
                                        &channelDesc,
                                        width,
                                        height));
        checkCudaErrors(cudaMemcpyToArray(cuArray,
                                          0,
                                          0,
                                          hData,
                                          size,
                                          cudaMemcpyHostToDevice));

        // Set texture parameters
        tex.addressMode[0] = cudaAddressModeBorder;
        tex.addressMode[1] = cudaAddressModeBorder;
        tex.filterMode = cudaFilterModeLinear;

        // Amount of padding
        int padWidth = (int)2*floor(filter_width/2);
        int padHeight = (int)2*floor(filter_height/2);

        // Bind the array to the texture
        checkCudaErrors(cudaBindTextureToArray(tex, cuArray, channelDesc));

        //copy host filter data to constant memory
        cudaMemcpyToSymbol(d_filter_constant, h_filter, sizeof(float)*filter_width*filter_height);

        printf("Texture ");

        executeTextureConvolution(height, width, dData, padHeight, padWidth);

        // copy result from device to host
        checkCudaErrors(cudaMemcpy(hOutputData,
                                   dData,
                                   size,
                                   cudaMemcpyDeviceToHost));

        // free memory
        checkCudaErrors(cudaFree(dData));
        checkCudaErrors(cudaFreeArray(cuArray));
        name = (char *)"Texture";

    }else if(option >= 1){

        // padded image
        float *hDataPadded = NULL;

        // Amount of padding
        int padWidth = (int)2*floor(filter_width/2);
        int padHeight = (int)2*floor(filter_height/2);

        // Allocate memory for padded images
        hDataPadded = (float *)malloc(sizeof(float)*(width+(padWidth))*(height+(padHeight)));


        // populate padded image with hData
        padded_populate(hDataPadded, hData, width, height/*, h_filter*/);

        if(option>1){

            //this is gpu related kernel initialization

            // Allocate device memory for result
            float *dData = NULL;
            checkCudaErrors(cudaMalloc((void **) &dData, size));

            // Allocate device memory for padded image
            float* d_padded_image = NULL;
            checkCudaErrors(cudaMalloc((void **)&d_padded_image, sizeof(float)*(width+(padWidth))*(height+(padHeight))));
            checkCudaErrors(cudaMemcpy(d_padded_image, hDataPadded, sizeof(float)*(width+(padWidth))*(height+(padHeight)), cudaMemcpyHostToDevice));


            if(option == 2){
                // Allocate device memory for filter
                float* d_filter = NULL;
                checkCudaErrors(cudaMalloc((void **)&d_filter, sizeof(float)*filter_width*filter_height));
                checkCudaErrors(cudaMemcpy(d_filter, h_filter, sizeof(float)*filter_width*filter_height, cudaMemcpyHostToDevice));
                printf("Naive ");
                //execute chosen kernel
                executeNaiveConvolution(height, width, d_padded_image, dData, d_filter, padWidth);
                //printf("Exectuted naive convolution \n");
                name = (char *)"Naive";
                //free memory
                checkCudaErrors(cudaFree(d_filter));
            }else if(option == 3){
                float* d_filter = NULL;
                checkCudaErrors(cudaMalloc((void **)&d_filter, sizeof(float)*filter_width*filter_height));
                checkCudaErrors(cudaMemcpy(d_filter, h_filter, sizeof(float)*filter_width*filter_height, cudaMemcpyHostToDevice));
                printf("Shared ");
                //execute chosen kernel
                executeSharedConvolution(height, width, d_padded_image, dData, d_filter, padWidth);
                //printf("Exectuted shared convolution \n");
                name = (char *)"Shared";
                //free memory
                checkCudaErrors(cudaFree(d_filter));
            }else if(option == 4){

                //copy host filter data to constant memory
                cudaMemcpyToSymbol(d_filter_constant, h_filter, sizeof(float)*filter_width*filter_height);
                printf("Constant Shared ");
                //execute chosen kernel
                executeConstantSharedConvolution(height, width, d_padded_image, dData, padWidth);
                name = (char *)"Constant_Shared";
                //printf("Exectuted constant shared convolution \n");
            }

            // copy result from device to host
            checkCudaErrors(cudaMemcpy(hOutputData,
                                       dData,
                                       size,
                                       cudaMemcpyDeviceToHost));

            // free memory
            checkCudaErrors(cudaFree(dData));
            checkCudaErrors(cudaFree(d_padded_image));
        }else if(option == 1){
            //cpu convolution
            printf("CPU ");
            cpuConvolution(hDataPadded, hOutputData, h_filter, width, height, padWidth/2, padHeight/2);
            name = (char *)"CPU";
            //printf("Exectuted cpu convolution \n");
        }else{
            //print ERROR
            perror("Invalid option");
        }

        //free memory
        free(hDataPadded);

    }else{
        //print ERROR
        perror("Invalid option");
    }

    // Write result to file
    char outputFilename[1024];
    strcpy(outputFilename, imagePath);
    strcpy(outputFilename + strlen(imagePath) - 4, "_");
    strcpy(outputFilename + strlen(imagePath) - 4 + 1, name);
    strcpy(outputFilename + strlen(imagePath) - 4 + 1 + strlen(name), "_out.pgm");
    sdkSavePGM(outputFilename, hOutputData, width, height);
    //printf("Wrote '%s'\n", outputFilename);

    //free space
    free(imagePath);
    free(h_filter);
    free(hData);
    free(hOutputData);
}

// generate filter for image
void filter_gen(float* filter, int width, int height){
  for(int i = 0; i < width*height; i++){
    filter[i] = (float)1/(width*height);
  }
  //filter[(int)(floor(width/2) + floor(height/2)*width)] = (float)(-1)*(4/(width*height));
}

// populate padded images
void padded_populate(float* padded, float* image, int width, int height/*, float *filter*/){
    // padded border increase
    int w = (int)floor(filter_width/2);
    int h = (int)floor(filter_height/2);

    // initialize padded array to 0
    for(int i = 0; i<(height+(2*h));i++){
      for(int j = 0; j<(width+(2*w)); j++){
        padded[j+(i*(width+(2*w)))] = (float)0;
      }
    }
    //populate padded array with image
    for(int i = 0; i<height;i++){
      for(int j = 0; j<width; j++){;
          padded[(j+w)+(i+h)*(width+2*w)] = image[j+(i*width)];
      }
    }
}

void executeNaiveConvolution(int height, int width, float *d_padded_image, float *dData, float *d_filter, int padWidth){

    dim3 dimBlock(32, 32, 1);
    dim3 dimGrid(width / dimBlock.x, height / dimBlock.y, 1);

    // Execute the kernel warmup
    naiveConvolutionKernel<<<dimGrid, dimBlock, 0>>>(d_padded_image, dData, d_filter, width, padWidth/2);

    checkCudaErrors(cudaDeviceSynchronize());

    StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    // Execute the kernel
    naiveConvolutionKernel<<<dimGrid, dimBlock, 0>>>(d_padded_image, dData, d_filter, width, padWidth/2);

    // Check if kernel execution generated an error
    getLastCudaError("Kernel execution failed");

    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&timer);
    printf("Processing time: %f (ms)\n", sdkGetTimerValue(&timer));
    //printf("%.2f Mpixels/sec\n",
           //(width *height / (sdkGetTimerValue(&timer) / 1000.0f)) / 1e6);
    sdkDeleteTimer(&timer);

}

void executeSharedConvolution(int height, int width, float *d_padded_image, float *dData, float *d_filter, int padWidth){

    dim3 dimBlock((TILE_WIDTH - (padWidth)), (TILE_WIDTH - (padWidth)), 1);
    dim3 dimGrid(width / dimBlock.x, height / dimBlock.y, 1);

    // Execute the kernel warmup
    sharedConvolutionKernel<<<dimGrid, dimBlock, 0>>>(d_padded_image, dData, d_filter, width, padWidth/2);

    checkCudaErrors(cudaDeviceSynchronize());

    StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    // Execute the kernel
    sharedConvolutionKernel<<<dimGrid, dimBlock, 0>>>(d_padded_image, dData, d_filter, width, padWidth/2);

    // Check if kernel execution generated an error
    getLastCudaError("Kernel execution failed");

    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&timer);
    printf("Processing time: %f (ms)\n", sdkGetTimerValue(&timer));
    //printf("%.2f Mpixels/sec\n",
           //(width *height / (sdkGetTimerValue(&timer) / 1000.0f)) / 1e6);
    sdkDeleteTimer(&timer);

}

void executeConstantSharedConvolution(int height, int width, float *d_padded_image, float *dData, int padWidth){

    dim3 dimBlock((TILE_WIDTH - (padWidth)), (TILE_WIDTH - (padWidth)), 1);
    dim3 dimGrid(width / dimBlock.x, height / dimBlock.y, 1);

    // Execute the kernel warmup
    constantSharedConvolutionKernel<<<dimGrid, dimBlock, 0>>>(d_padded_image, dData, width, padWidth/2);

    checkCudaErrors(cudaDeviceSynchronize());

    StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    // Execute the kernel
    constantSharedConvolutionKernel<<<dimGrid, dimBlock, 0>>>(d_padded_image, dData, width, padWidth/2);

    // Check if kernel execution generated an error
    getLastCudaError("Kernel execution failed");

    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&timer);
    printf("Processing time: %f (ms)\n", sdkGetTimerValue(&timer));
    //printf("%.2f Mpixels/sec\n",
           //(width *height / (sdkGetTimerValue(&timer) / 1000.0f)) / 1e6);
    sdkDeleteTimer(&timer);
}

void executeTextureConvolution(int height, int width, float *dData, int padHeight, int padWidth){

    dim3 dimBlock(32, 32, 1);
    dim3 dimGrid(width / dimBlock.x, height / dimBlock.y, 1);

    // Execute the kernel warmup
    textureConvolutionKernel<<<dimGrid, dimBlock, 0>>>(dData, padWidth/2, padHeight/2, width);

    checkCudaErrors(cudaDeviceSynchronize());

    StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    // Execute the kernel
    textureConvolutionKernel<<<dimGrid, dimBlock, 0>>>(dData, padWidth/2, padHeight/2, width);

    // Check if kernel execution generated an error
    getLastCudaError("Kernel execution failed");

    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&timer);
    printf("Processing time: %f (ms)\n", sdkGetTimerValue(&timer));
    //printf("%.2f Mpixels/sec\n",
           //(width *height / (sdkGetTimerValue(&timer) / 1000.0f)) / 1e6);
    sdkDeleteTimer(&timer);

}
