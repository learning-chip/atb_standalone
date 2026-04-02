// Shim header: kernel sources in this repo expect `lib/matmul_intf.h`,
// while CANN provides it under `adv_api/matmul/matmul_intf.h`.
//
// Keep this shim local so the standalone build doesn't depend on ops-transformer include layout.

#pragma once

#include "adv_api/matmul/matmul_intf.h"

