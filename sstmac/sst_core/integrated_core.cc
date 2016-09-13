
#include <sstmac/common/sstmac_config.h>
#include <sstmac/main/sstmac.h>

#include <sst/core/element.h>
#include <sstmac/sst_core/integrated_core.h>
#include <sstmac/sst_core/connectable_wrapper.h>
#include <sstmac/hardware/interconnect/interconnect.h>
#include <sstmac/hardware/packet_flow/packet_flow_nic.h>
#include <sstmac/hardware/packet_flow/packet_flow_switch.h>
#include <sstmac/hardware/node/simple_node.h>

#include <vector>

#include <sprockit/util.h>
#include <sprockit/unordered.h>
#include <sprockit/debug.h>
#include <sprockit/output.h>

#include <Python.h>

#include <stdio.h>
#include <stddef.h>

static char py_sstmacro[] = {
#include "sstmacro.inc"
    0x00};

using namespace sstmac;
using namespace SST;

void init_python_topology(PyObject* module);


PyObject*
py_array_from_int_vector(const std::vector<int>& vec)
{
  PyObject* pVec = PyTuple_New(vec.size());
  for (int i = 0; i < vec.size(); ++i) {
       PyObject* pValue = PyInt_FromLong(vec[i]);
       if (!pValue) {
         spkt_abort_printf("cannot convert array value for building PyTuple");
       }
       PyTuple_SetItem(pVec, i, pValue);
  }
  return pVec;
}

void
int_vector_from_py_array(PyObject* tuple, std::vector<int>& vec)
{
  Py_ssize_t size = PyTuple_Size(tuple);
  vec.resize(size);
  for (int i=0; i < size; ++i){
    PyObject* obj = PyTuple_GetItem(tuple,i);
    int item  = PyInt_AsLong(obj);
    vec[i] = item;
  }
}

PyObject*
set_debug_flags(PyObject* self, PyObject* args)
{
  PyObject* flags = PyTuple_GetItem(args, 0);
  Py_ssize_t size = PyTuple_Size(flags);
  for (int i=0; i < size; ++i){
    PyObject* obj = PyTuple_GetItem(flags,i);
    const char* str = PyString_AsString(obj);
    sprockit::debug::turn_on(str);
  }
  Py_RETURN_NONE;
}

PyObject*
init(PyObject* self, PyObject* args)
{
  sprockit::output::init_out0(&std::cout);
  sprockit::output::init_err0(&std::cerr);
  sprockit::output::init_outn(&std::cout);
  sprockit::output::init_errn(&std::cerr);
  Py_RETURN_NONE;
}

static void
add_params(PyObject* dict, sprockit::sim_parameters* params)
{
  sprockit::sim_parameters::key_value_map::iterator it, end = params->end();
  for (it=params->begin(); it != end; ++it){
    const std::string& key_name = it->first;
    const std::string& key_value = it->second.value;
    PyObject* key = PyString_FromString(key_name.c_str());
    PyObject* val = PyString_FromString(key_value.c_str());
    PyDict_SetItem(dict, key, val);
  }
}

static void
add_sub_params(PyObject* dict, sprockit::sim_parameters* params)
{
  sprockit::sim_parameters::namespace_iterator it, end = params->ns_end();
  for (it=params->ns_begin(); it != end; ++it){
    sprockit::sim_parameters* subparams = it->second;
    PyObject* subdict = PyDict_New();
    const char* key = it->first.c_str();
    PyDict_SetItemString(dict, key, subdict);
    add_params(subdict, subparams);
    add_sub_params(subdict, subparams);
  }
}

static PyObject*
read_params(PyObject* self, PyObject* args)
{
  PyObject* sys_argv = PyTuple_GetItem(args, 0);
  int argc = PyList_Size(sys_argv);
  char** argv = new char*[argc];
  for (int i=0; i < argc; ++i){
    PyObject* obj = PyList_GetItem(sys_argv,i);
    const char* str = PyString_AsString(obj);
    argv[i] = const_cast<char*>(str);
  }

  sprockit::sim_parameters params;
  sstmac::try_main(&params, argc, argv, true/*only params*/);

  PyObject* dict = PyDict_New();
  add_params(dict, &params);
  add_sub_params(dict, &params);

  delete[] argv;

  return dict;
}


static const ElementInfoComponent macro_components[] = {
    sstmac::hw::packet_flow_switch_element_info,
    sstmac::hw::simple_node_element_info,
    {NULL, NULL, NULL, NULL}
};


static PyMethodDef sst_macro_integrated_methods[] = {
    { "readParams", read_params, METH_VARARGS, "parse command line options and read parameters" },
    { "debug", set_debug_flags, METH_VARARGS, "set debug flags" },
    { "init", init, METH_VARARGS, "initialize the macro environment" },
    { NULL, NULL, 0, NULL }
};

static void* gen_sst_macro_integrated_pymodule(void)
{
  PyObject* tmpModule = Py_InitModule("sstmac", sst_macro_integrated_methods);
  PyObject *code = Py_CompileString(py_sstmacro, "sstmacro", Py_file_input);
  PyObject* module = PyImport_ExecCodeModule("sst.macro", code);
  init_python_topology(tmpModule);
  sstmac::connectable_proxy_component::sst = PyImport_ImportModule("sst");
  return module;
}

extern "C" {

ElementLibraryInfo macro_eli = {
    "macro",
    "SST Macroscale integrated components",
    macro_components,                  // Components
    NULL,                              // Events
    NULL,                              // Introspectors
    NULL,                              // Modules
    NULL,                              // Partitioners
    NULL,                              // Generators
    gen_sst_macro_integrated_pymodule  // Python Module Generator
};


} // end extern "C"

