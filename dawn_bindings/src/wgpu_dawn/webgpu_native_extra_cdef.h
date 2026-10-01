typedef struct WGPUSurfaceSourceXlibWindow {
    WGPUChainedStruct chain;
    void * display;
    uint64_t window;
} WGPUSurfaceSourceXlibWindow ;
typedef struct WGPUSurfaceSourceWaylandSurface {
    WGPUChainedStruct chain;
    void * display;
    void * surface;
} WGPUSurfaceSourceWaylandSurface ;
typedef struct WGPUSurfaceSourceXCBWindow {
    WGPUChainedStruct chain;
    void * connection;
    uint32_t window;
} WGPUSurfaceSourceXCBWindow ;
typedef struct WGPUSurfaceSourceMetalLayer {
    WGPUChainedStruct chain;
    void * layer;
} WGPUSurfaceSourceMetalLayer ;
typedef struct WGPUSurfaceSourceWindowsHWND {
    WGPUChainedStruct chain;
    void * hinstance;
    void * hwnd;
} WGPUSurfaceSourceWindowsHWND ;
