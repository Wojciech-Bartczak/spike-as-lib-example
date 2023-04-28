project('spike-lib-test', ['c', 'cpp'])

spike_dep = dependency('riscv-riscv')
disasm_dep = dependency('riscv-disasm')
fmt_dep = dependency('fmt')

src = ['main.cc']

executable('simple-main', sources: src,
           dependencies: [spike_dep, disasm_dep, fmt_dep],
           cpp_args: ['-Wall', '-O2', '-g', '-std=c++17'])