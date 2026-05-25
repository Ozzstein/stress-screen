from stress_screen.topology import derive_topology
import pytest

def test_4p8s_derivation():
    topo = derive_topology(48, 6)
    assert topo.config_name == "4P8S"
    assert topo.series == 8
    assert topo.parallel == 4
    assert topo.module_count == 6
    assert topo.channels_in_module(1) == list(range(0, 8))
    assert topo.channels_in_module(6) == list(range(40, 48))
    assert topo.group_index_in_module(0) == 1
    assert topo.group_index_in_module(7) == 8
    assert topo.temp_sensors_for_group(1, 1) == [1]
    assert topo.temp_sensors_for_group(1, 2) == [1, 2]
    assert topo.temp_sensors_for_group(1, 8) == [7]

def test_2p16s_derivation():
    topo = derive_topology(64, 4)
    assert topo.config_name == "2P16S"
    assert topo.series == 16

def test_1p32s_derivation():
    topo = derive_topology(32, 1)
    assert topo.config_name == "1P32S"
    assert topo.series == 32

def test_invalid_non_integer_groups():
    with pytest.raises(ValueError, match="not an integer"):
        derive_topology(50, 3)

def test_invalid_zero_module_count():
    with pytest.raises(ValueError):
        derive_topology(48, 0)
