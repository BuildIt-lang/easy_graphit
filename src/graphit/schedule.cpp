#include "graphit/schedule.h"
#include "graphit/graphit_types.h"
namespace graphit{

int SimpleGPUSchedule::default_cta_size = 512;
int SimpleGPUSchedule::default_max_cta = 60;


void HybridGPUSchedule::bindThreshold(dyn_var<float> &t) {
	static_threshold = -1;
	threshold = t.addr();	
}
}
