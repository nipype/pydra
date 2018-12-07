# -*- coding: utf-8 -*-

import typing as ty
import os
import pytest

from ..task import (to_task, PrintMessenger, FileMessenger,
                    AuditFlag)

def test_annotated_func():
    @to_task
    def testfunc(a: int, b: float = 0.1) -> ty.NamedTuple('Output',
                                                          [('out1', float)]):
        return a + b

    funky = testfunc(a=1)
    assert hasattr(funky.inputs, 'a')
    assert hasattr(funky.inputs, 'b')
    assert hasattr(funky.inputs, '_func')
    assert getattr(funky.inputs, 'a') == 1
    assert getattr(funky.inputs, 'b') == 0.1
    assert getattr(funky.inputs, '_func') is not None
    assert set(funky.output_names) == set(['out1'])
    assert funky.inputs.hash == '17772c3aec9540a8dd3e187eecd2301a09c9a25c6e371ddd86e31e3a1ecfeefa'
    assert funky.inputs.hash == funky.checksum

    result = funky()
    assert hasattr(result, 'output')
    assert hasattr(result.output, 'out1')
    assert result.output.out1 == 1.1

    assert os.path.exists(funky.cache_dir / funky.checksum / '_result.pklz')

    funky.result()  # should not recompute

    funky.inputs.a = 2
    assert funky.checksum == '537d25885fd2ea5662b7701ba02c132c52a9078a3a2d56aa903a777ea90e5536'
    assert funky.result() is None
    funky()
    result = funky.result()
    assert result.output.out1 == 2.1


def test_notannotated_func():
    @to_task
    def no_annots(c, d):
        return c + d


    natask = no_annots(c=17, d=3.2)
    assert hasattr(natask.inputs, 'c')
    assert hasattr(natask.inputs, 'd')
    assert hasattr(natask.inputs, '_func')

    result = natask.run()
    assert hasattr(result, 'output')
    assert hasattr(result.output, 'out')
    assert result.output.out == 20.2


def test_audit(tmpdir):
    @to_task
    def testfunc(a: int, b: float = 0.1) -> ty.NamedTuple('Output',
                                                          [('out', float)]):
        return a + b

    funky = testfunc(a=1, audit_flags=AuditFlag.PROV, messengers=PrintMessenger())
    result = funky()

    funky = testfunc(a=1, audit_flags=AuditFlag.PROV, messengers=FileMessenger())
    message_path = tmpdir / funky.checksum / 'messages'
    funky.cache_dir = tmpdir
    funky.messenger_args=dict(message_dir=message_path)
    funky()
    from glob import glob
    assert len(glob(str(message_path / '*.jsonld'))) == 2
