/* SPDX-License-Identifier: BSD-2-Clause
 * Copyright(C) 2023 Marvell.
 */

#include <cstdint>
#include <cstdio>
#include <algorithm>
#include <iostream>
#include <map>
#include <optional>
#include <utility>
#include <vector>

#include <sys/types.h>
#include <signal.h>
#include <endian.h>

#include <fmt/core.h>
#include <riscv/cfg.h>
#include <riscv/devices.h>
#include <riscv/processor.h>
#include <riscv/simif.h>
#include <riscv/disasm.h>
#include <riscv/decode.h>

#define zext(x, pos) (((reg_t)(x) << (64 - (pos))) >> (64 - (pos)))

#define PROG_STEPS  17

static const uint8_t simple_prog[] = {
    0x19, 0xa0,              // j handle_reset
    0x40, 0x03, 0x6f, 0x00,  // j handle_trap
    0xf3, 0x22, 0x50, 0x30,  // csrr t0,mtvect
    0x0d, 0x43,              // li t1, 3
    0xb3, 0xf2, 0x62, 0x00,  // and t0, t0, t1
    0x63, 0x9b, 0x02, 0x00,  // bnez t0, 40000026
    0x97, 0x02, 0x00, 0x00,   // auipc t0, 0
    0x93, 0x82, 0xe2, 0xfe,  // add t0,t0,-18
    0x73, 0x90, 0x52, 0x30,  // csrw mtvec,t0
    0x73, 0x10, 0x00, 0x30,  // csrw mstatus,zero
    0x11, 0xa0,              // j 40000028
    0x01, 0xa0,              // j 40000026
    0x97, 0x11, 0x00, 0x00,  // auipc gp,0x1
    0x93, 0x81, 0x21, 0x81,  // add gp,gp,-2030
    0x73, 0x00, 0x50, 0x10,  // wfi
    0xf5, 0xbf,              // j 40000030
    0x6f, 0x00, 0x00, 0x00   // j 40000036
};

class simple_cpu : public simif_t
{
    static const reg_t IMEM_BASE = 0x40000000;
    static const reg_t MAX_PADDR_BITS = 56;

    cfg_t m_cfg;
    isa_parser_t isa;
    processor_t *core;
    bus_t bus;
    std::map<size_t, processor_t *> harts;
    rom_device_t *rom;
    std::vector<char> rom_data;

    bool paddr_ok(reg_t addr)
    {
        return (addr >> MAX_PADDR_BITS) == 0;
    }

public:
    simple_cpu();
    ~simple_cpu();

    void run(size_t steps = 1);

    /* simif */
    char* addr_to_mem(reg_t paddr) final;

    bool reservable(reg_t paddr) final {
        return addr_to_mem(paddr);
    }

    bool mmio_fetch(reg_t paddr, size_t len, uint8_t* bytes) final {
       return mmio_load(paddr, len, bytes);
    }

    bool mmio_load(reg_t paddr, size_t len, uint8_t* bytes) final;
    bool mmio_store(reg_t paddr, size_t len, const uint8_t* bytes) final;

    // Callback for processors to let the simulation know they were reset.
    void proc_reset(unsigned id) final {
    }

    const cfg_t &get_cfg() const final {
      return m_cfg;
    }
    const std::map<size_t, processor_t*>& get_harts() const final {
        return harts;
    }

    const char* get_symbol(uint64_t paddr) final {
        return NULL;
    }

    unsigned nprocs() const {
        return 1;
    }

    void show_regs() const;
};

simple_cpu::simple_cpu()
  : m_cfg(std::pair<reg_t, reg_t>(),
          nullptr,
          "RV32IMACZba_Zbb_Zbc_Zbs_Zicsr",
          "MSU",
          "vlen:32,elen:32",
          true,
          endianness_t::endianness_little,
          0,
          std::vector<mem_cfg_t>(),
          std::vector<size_t>(),
          false,
          static_cast<reg_t>(0)),
    isa("RV32IMACZba_Zbb_Zbc_Zbs_Zicsr", "m"),
    core(nullptr),
    bus(),
    harts(),
    rom(nullptr),
    rom_data(std::begin(simple_prog), std::end(simple_prog))
{
    rom = new rom_device_t(rom_data);
    core = new processor_t(&isa, &m_cfg, this, 0, false, stdout, std::cout);
    core->get_state()->pc = simple_cpu::IMEM_BASE;
    harts.insert(std::make_pair(0, core));
    // This core runs only in debug
    core->set_debug(true);

    bus.add_device(simple_cpu::IMEM_BASE, rom);
}

simple_cpu::~simple_cpu()
{
    delete core;
    delete rom;
}

void simple_cpu::run(size_t steps)
{
    core->step(steps);
}

bool simple_cpu::mmio_load(reg_t paddr, size_t len, uint8_t* bytes)
{
  if (paddr + len < paddr || !paddr_ok(paddr + len - 1))
    return false;
  return bus.load(paddr, len, bytes);
}

bool simple_cpu::mmio_store(reg_t paddr, size_t len, const uint8_t* bytes)
{
  if (paddr + len < paddr || !paddr_ok(paddr + len - 1))
    return false;
  return bus.store(paddr, len, bytes);
}

char * simple_cpu::addr_to_mem(reg_t addr)
{
	auto desc = bus.find_device(addr);
	if (auto mem = dynamic_cast<mem_t *>(desc.second)) {
		if (addr - desc.first < mem->size()) {
			return mem->contents(addr - desc.first);
		}
	}
	return NULL;
}

void simple_cpu::show_regs() const
{
	/* Show all the regs */
	for (int r = 0; r < NXPR; ++r) {
        fmt::print("{:4s} {:#010x} ", xpr_name[r], zext(core->get_state()->XPR[r], 32));
        if ((r + 1) % 4 == 0)
            fmt::print("\n");
	}
}

static int done = 0;

void sig_handler(int signo)
{
    if (signo == SIGINT || signo == SIGTERM)
        done = 1;
}

int main(int argc, char *argv[])
{
    struct sigaction sa;
    int ret = 0;
    size_t step = 0;

    sa.sa_flags = 0;
    sigemptyset(&sa.sa_mask);
    sa.sa_handler = &sig_handler;

    ret = sigaction(SIGINT, &sa, NULL);
    if (ret < 0) {
        return 1;
    }

    ret = sigaction(SIGTERM, &sa, NULL);
    if (ret < 0) {
        return 1;
    }

    simple_cpu spam;

    while (!done) {
        if (step >= PROG_STEPS)
            break;

        fmt::print("--\n");
        spam.run();
        // Show all regs
        spam.show_regs();
        fmt::print("--\n");
        step++;
    }


    fmt::print("Done. Bye!\n");
    return 0;
}