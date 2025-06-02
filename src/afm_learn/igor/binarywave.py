# This code is derived from the library available at https://github.com/wking/igor, 
# which is licensed under the GNU Lesser General Public License (LGPL) version 3.

# GNU LESSER GENERAL PUBLIC LICENSE
#                        Version 3, 29 June 2007

#  Copyright (C) 2007 Free Software Foundation, Inc. <http://fsf.org/>
#  Everyone is permitted to copy and distribute verbatim copies
#  of this license document, but changing it is not allowed.


#   This version of the GNU Lesser General Public License incorporates
# the terms and conditions of version 3 of the GNU General Public
# License, supplemented by the additional permissions listed below.

#   0. Additional Definitions. 

#   As used herein, "this License" refers to version 3 of the GNU Lesser
# General Public License, and the "GNU GPL" refers to version 3 of the GNU
# General Public License.

#   "The Library" refers to a covered work governed by this License,
# other than an Application or a Combined Work as defined below.

#   An "Application" is any work that makes use of an interface provided
# by the Library, but which is not otherwise based on the Library.
# Defining a subclass of a class defined by the Library is deemed a mode
# of using an interface provided by the Library.

#   A "Combined Work" is a work produced by combining or linking an
# Application with the Library.  The particular version of the Library
# with which the Combined Work was made is also called the "Linked
# Version".

#   The "Minimal Corresponding Source" for a Combined Work means the
# Corresponding Source for the Combined Work, excluding any source code
# for portions of the Combined Work that, considered in isolation, are
# based on the Application, and not on the Linked Version.

#   The "Corresponding Application Code" for a Combined Work means the
# object code and/or source code for the Application, including any data
# and utility programs needed for reproducing the Combined Work from the
# Application, but excluding the System Libraries of the Combined Work.

#   1. Exception to Section 3 of the GNU GPL.

#   You may convey a covered work under sections 3 and 4 of this License
# without being bound by section 3 of the GNU GPL.

#   2. Conveying Modified Versions.

#   If you modify a copy of the Library, and, in your modifications, a
# facility refers to a function or data to be supplied by an Application
# that uses the facility (other than as an argument passed when the
# facility is invoked), then you may convey a copy of the modified
# version:

#    a) under this License, provided that you make a good faith effort to
#    ensure that, in the event an Application does not supply the
#    function or data, the facility still operates, and performs
#    whatever part of its purpose remains meaningful, or

#    b) under the GNU GPL, with none of the additional permissions of
#    this License applicable to that copy.

#   3. Object Code Incorporating Material from Library Header Files.

#   The object code form of an Application may incorporate material from
# a header file that is part of the Library.  You may convey such object
# code under terms of your choice, provided that, if the incorporated
# material is not limited to numerical parameters, data structure
# layouts and accessors, or small macros, inline functions and templates
# (ten or fewer lines in length), you do both of the following:

#    a) Give prominent notice with each copy of the object code that the
#    Library is used in it and that the Library and its use are
#    covered by this License.

#    b) Accompany the object code with a copy of the GNU GPL and this license
#    document.

#   4. Combined Works.

#   You may convey a Combined Work under terms of your choice that,
# taken together, effectively do not restrict modification of the
# portions of the Library contained in the Combined Work and reverse
# engineering for debugging such modifications, if you also do each of
# the following:

#    a) Give prominent notice with each copy of the Combined Work that
#    the Library is used in it and that the Library and its use are
#    covered by this License.

#    b) Accompany the Combined Work with a copy of the GNU GPL and this license
#    document.

#    c) For a Combined Work that displays copyright notices during
#    execution, include the copyright notice for the Library among
#    these notices, as well as a reference directing the user to the
#    copies of the GNU GPL and this license document.

#    d) Do one of the following:

#        0) Convey the Minimal Corresponding Source under the terms of this
#        License, and the Corresponding Application Code in a form
#        suitable for, and under terms that permit, the user to
#        recombine or relink the Application with a modified version of
#        the Linked Version to produce a modified Combined Work, in the
#        manner specified by section 6 of the GNU GPL for conveying
#        Corresponding Source.

#        1) Use a suitable shared library mechanism for linking with the
#        Library.  A suitable mechanism is one that (a) uses at run time
#        a copy of the Library already present on the user's computer
#        system, and (b) will operate properly with a modified version
#        of the Library that is interface-compatible with the Linked
#        Version. 

#    e) Provide Installation Information, but only if you would otherwise
#    be required to provide such information under section 6 of the
#    GNU GPL, and only to the extent that such information is
#    necessary to install and execute a modified version of the
#    Combined Work produced by recombining or relinking the
#    Application with a modified version of the Linked Version. (If
#    you use option 4d0, the Installation Information must accompany
#    the Minimal Corresponding Source and Corresponding Application
#    Code. If you use option 4d1, you must provide the Installation
#    Information in the manner specified by section 6 of the GNU GPL
#    for conveying Corresponding Source.)

#   5. Combined Libraries.

#   You may place library facilities that are a work based on the
# Library side by side in a single library together with other library
# facilities that are not Applications and are not covered by this
# License, and convey such a combined library under terms of your
# choice, if you do both of the following:

#    a) Accompany the combined library with a copy of the same work based
#    on the Library, uncombined with any other library facilities,
#    conveyed under the terms of this License.

#    b) Give prominent notice with the combined library that part of it
#    is a work based on the Library, and explaining where to find the
#    accompanying uncombined form of the same work.

#   6. Revised Versions of the GNU Lesser General Public License.

#   The Free Software Foundation may publish revised and/or new versions
# of the GNU Lesser General Public License from time to time. Such new
# versions will be similar in spirit to the present version, but may
# differ in detail to address new problems or concerns.

#   Each version is given a distinguishing version number. If the
# Library as you received it specifies that a certain numbered version
# of the GNU Lesser General Public License "or any later version"
# applies to it, you have the option of following the terms and
# conditions either of that published version or of any later version
# published by the Free Software Foundation. If the Library as you
# received it does not specify a version number of the GNU Lesser
# General Public License, you may choose any version of the GNU Lesser
# General Public License ever published by the Free Software Foundation.

#   If the Library as you received it specifies that a proxy can decide
# whether future versions of the GNU Lesser General Public License shall
# apply, that proxy's public statement of acceptance of any version is
# permanent authorization for you to choose that version for the
# Library.


# Copyright (C) 2010-2012 W. Trevor King <wking@tremily.us>
#
# This file is part of igor.
#
# igor is free software: you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# igor is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with igor.  If not, see <http://www.gnu.org/licenses/>.

# "Read IGOR Binary Wave files into Numpy arrays."

# Based on WaveMetric's Technical Note 003, "Igor Binary Format"
#   ftp://ftp.wavemetrics.net/IgorPro/Technical_Notes/TN003.zip
# From ftp://ftp.wavemetrics.net/IgorPro/Technical_Notes/TN000.txt
#   We place no restrictions on copying Technical Notes, with the
#   exception that you cannot resell them. So read, enjoy, and
#   share. We hope IGOR Technical Notes will provide you with lots of
#   valuable information while you are developing IGOR applications.



from __future__ import absolute_import
import array as _array
import struct as _struct
import sys as _sys
import types as _types

import numpy as _numpy

from igor import LOG as _LOG
from igor.struct import Structure as _Structure
from igor.struct import DynamicStructure as _DynamicStructure
from igor.struct import Field as _Field
from igor.struct import DynamicField as _DynamicField
from igor.util import assert_null as _assert_null
from igor.util import byte_order as _byte_order
from igor.util import need_to_reorder_bytes as _need_to_reorder_bytes
from igor.util import checksum as _checksum


# Numpy doesn't support complex integers by default, see
#   http://mail.python.org/pipermail/python-dev/2002-April/022408.html
#   http://mail.scipy.org/pipermail/numpy-discussion/2007-October/029447.html
# So we roll our own types.  See
#   http://docs.scipy.org/doc/numpy/user/basics.rec.html
#   http://docs.scipy.org/doc/numpy/reference/generated/numpy.dtype.html
complexInt8 = _numpy.dtype([('real', _numpy.int8), ('imag', _numpy.int8)])
complexInt16 = _numpy.dtype([('real', _numpy.int16), ('imag', _numpy.int16)])
complexInt32 = _numpy.dtype([('real', _numpy.int32), ('imag', _numpy.int32)])
complexUInt8 = _numpy.dtype([('real', _numpy.uint8), ('imag', _numpy.uint8)])
complexUInt16 = _numpy.dtype(
    [('real', _numpy.uint16), ('imag', _numpy.uint16)])
complexUInt32 = _numpy.dtype(
    [('real', _numpy.uint32), ('imag', _numpy.uint32)])


class StaticStringField (_DynamicField):
    _null_terminated = False
    _array_size_field = None
    def __init__(self, *args, **kwargs):
        if 'array' not in kwargs:
            kwargs['array'] = True
        super(StaticStringField, self).__init__(*args, **kwargs)

    def post_unpack(self, parents, data):
        wave_structure = parents[-1]
        wave_data = self._get_structure_data(parents, data, wave_structure)
        d = self._normalize_string(wave_data[self.name])
        wave_data[self.name] = d

    def _normalize_string(self, d):
        if isinstance(d, bytes):
            pass
        elif hasattr(d, 'tobytes'):
            d = d.tobytes()
        elif hasattr(d, 'tostring'):  # Python 2 compatibility
            d = d.tostring()
        else:
            d = b''.join(d)
        if self._array_size_field:
            start = 0
            strings = []
            for count in self.counts:
                end = start + count
                if end > start:
                    strings.append(d[start:end])
                    if self._null_terminated:
                        strings[-1] = strings[-1].split(b'\x00', 1)[0]
                    start = end
        elif self._null_terminated:
            d = d.split(b'\x00', 1)[0]
        return d


class NullStaticStringField (StaticStringField):
    _null_terminated = True


# Begin IGOR constants and typedefs from IgorBin.h

# From IgorMath.h
TYPE_TABLE = {        # (key: integer flag, value: numpy dtype)
    0:None,           # Text wave, not handled in ReadWave.c
    1:_numpy.complex128, # NT_CMPLX, makes number complex.
    2:_numpy.float32, # NT_FP32, 32 bit fp numbers.
    3:_numpy.complex64,
    4:_numpy.float64, # NT_FP64, 64 bit fp numbers.
    5:_numpy.complex128,
    8:_numpy.int8,    # NT_I8, 8 bit signed integer. Requires Igor Pro
                      # 2.0 or later.
    9:complexInt8,
    0x10:_numpy.int16,# NT_I16, 16 bit integer numbers. Requires Igor
                      # Pro 2.0 or later.
    0x11:complexInt16,
    0x20:_numpy.int32,# NT_I32, 32 bit integer numbers. Requires Igor
                      # Pro 2.0 or later.
    0x21:complexInt32,
#   0x40:None,        # NT_UNSIGNED, Makes above signed integers
#                     # unsigned. Requires Igor Pro 3.0 or later.
    0x48:_numpy.uint8,
    0x49:complexUInt8,
    0x50:_numpy.uint16,
    0x51:complexUInt16,
    0x60:_numpy.uint32,
    0x61:complexUInt32,
}

# From wave.h
MAXDIMS = 4

# From binary.h
BinHeader1 = _Structure(  # `version` field pulled out into Wave
    name='BinHeader1',
    fields=[
        _Field('l', 'wfmSize', help='The size of the WaveHeader2 data structure plus the wave data plus 16 bytes of padding.'),
        _Field('h', 'checksum', help='Checksum over this header and the wave header.'),
        ])

BinHeader2 = _Structure(  # `version` field pulled out into Wave
    name='BinHeader2',
    fields=[
        _Field('l', 'wfmSize', help='The size of the WaveHeader2 data structure plus the wave data plus 16 bytes of padding.'),
        _Field('l', 'noteSize', help='The size of the note text.'),
        _Field('l', 'pictSize', default=0, help='Reserved. Write zero. Ignore on read.'),
        _Field('h', 'checksum', help='Checksum over this header and the wave header.'),
        ])

BinHeader3 = _Structure(  # `version` field pulled out into Wave
    name='BinHeader3',
    fields=[
        _Field('l', 'wfmSize', help='The size of the WaveHeader2 data structure plus the wave data plus 16 bytes of padding.'),
        _Field('l', 'noteSize', help='The size of the note text.'),
        _Field('l', 'formulaSize', help='The size of the dependency formula, if any.'),
        _Field('l', 'pictSize', default=0, help='Reserved. Write zero. Ignore on read.'),
        _Field('h', 'checksum', help='Checksum over this header and the wave header.'),
        ])

BinHeader5 = _Structure(  # `version` field pulled out into Wave
    name='BinHeader5',
    fields=[
        _Field('h', 'checksum', help='Checksum over this header and the wave header.'),
        _Field('l', 'wfmSize', help='The size of the WaveHeader5 data structure plus the wave data.'),
        _Field('l', 'formulaSize', help='The size of the dependency formula, if any.'),
        _Field('l', 'noteSize', help='The size of the note text.'),
        _Field('l', 'dataEUnitsSize', help='The size of optional extended data units.'),
        _Field('l', 'dimEUnitsSize', help='The size of optional extended dimension units.', count=MAXDIMS, array=True),
        _Field('l', 'dimLabelsSize', help='The size of optional dimension labels.', count=MAXDIMS, array=True),
        _Field('l', 'sIndicesSize', help='The size of string indicies if this is a text wave.'),
        _Field('l', 'optionsSize1', default=0, help='Reserved. Write zero. Ignore on read.'),
        _Field('l', 'optionsSize2', default=0, help='Reserved. Write zero. Ignore on read.'),
        ])


# From wave.h
MAX_WAVE_NAME2 = 18 # Maximum length of wave name in version 1 and 2
                    # files. Does not include the trailing null.
MAX_WAVE_NAME5 = 31 # Maximum length of wave name in version 5
                    # files. Does not include the trailing null.
MAX_UNIT_CHARS = 3

# Header to an array of waveform data.

# `wData` field pulled out into DynamicWaveDataField1
WaveHeader2 = _DynamicStructure(
    name='WaveHeader2',
    fields=[
        _Field('h', 'type', help='See types (e.g. NT_FP64) above. Zero for text waves.'),
        _Field('P', 'next', default=0, help='Used in memory only. Write zero. Ignore on read.'),
        NullStaticStringField('c', 'bname', help='Name of wave plus trailing null.', count=MAX_WAVE_NAME2+2),
        _Field('h', 'whVersion', default=0, help='Write 0. Ignore on read.'),
        _Field('h', 'srcFldr', default=0, help='Used in memory only. Write zero. Ignore on read.'),
        _Field('P', 'fileName', default=0, help='Used in memory only. Write zero. Ignore on read.'),
        _Field('c', 'dataUnits', default=0, help='Natural data units go here - null if none.', count=MAX_UNIT_CHARS+1, array=True),
        _Field('c', 'xUnits', default=0, help='Natural x-axis units go here - null if none.', count=MAX_UNIT_CHARS+1, array=True),
        _Field('l', 'npnts', help='Number of data points in wave.'),
        _Field('h', 'aModified', default=0, help='Used in memory only. Write zero. Ignore on read.'),
        _Field('d', 'hsA', help='X value for point p = hsA*p + hsB'),
        _Field('d', 'hsB', help='X value for point p = hsA*p + hsB'),
        _Field('h', 'wModified', default=0, help='Used in memory only. Write zero. Ignore on read.'),
        _Field('h', 'swModified', default=0, help='Used in memory only. Write zero. Ignore on read.'),
        _Field('h', 'fsValid', help='True if full scale values have meaning.'),
        _Field('d', 'topFullScale', help='The min full scale value for wave.'), # sic, 'min' should probably be 'max'
        _Field('d', 'botFullScale', help='The min full scale value for wave.'),
        _Field('c', 'useBits', default=0, help='Used in memory only. Write zero. Ignore on read.'),
        _Field('c', 'kindBits', default=0, help='Reserved. Write zero. Ignore on read.'),
        _Field('P', 'formula', default=0, help='Used in memory only. Write zero. Ignore on read.'),
        _Field('l', 'depID', default=0, help='Used in memory only. Write zero. Ignore on read.'),
        _Field('L', 'creationDate', help='DateTime of creation.  Not used in version 1 files.'),
        _Field('c', 'wUnused', default=0, help='Reserved. Write zero. Ignore on read.', count=2, array=True),
        _Field('L', 'modDate', help='DateTime of last modification.'),
        _Field('P', 'waveNoteH', help='Used in memory only. Write zero. Ignore on read.'),
        ])

# `sIndices` pointer unset (use Wave5_data['sIndices'] instead).  This
# field is filled in by DynamicStringIndicesDataField.
# `wData` field pulled out into DynamicWaveDataField5
WaveHeader5 = _DynamicStructure(
    name='WaveHeader5',
    fields=[
        _Field('P', 'next', help='link to next wave in linked list.'),
        _Field('L', 'creationDate', help='DateTime of creation.'),
        _Field('L', 'modDate', help='DateTime of last modification.'),
        _Field('l', 'npnts', help='Total number of points (multiply dimensions up to first zero).'),
        _Field('h', 'type', help='See types (e.g. NT_FP64) above. Zero for text waves.'),
        _Field('h', 'dLock', default=0, help='Reserved. Write zero. Ignore on read.'),
        _Field('c', 'whpad1', default=0, help='Reserved. Write zero. Ignore on read.', count=6, array=True),
        _Field('h', 'whVersion', default=1, help='Write 1. Ignore on read.'),
        NullStaticStringField('c', 'bname', help='Name of wave plus trailing null.', count=MAX_WAVE_NAME5+1),
        _Field('l', 'whpad2', default=0, help='Reserved. Write zero. Ignore on read.'),
        _Field('P', 'dFolder', default=0, help='Used in memory only. Write zero. Ignore on read.'),
        # Dimensioning info. [0] == rows, [1] == cols etc
        _Field('l', 'nDim', help='Number of of items in a dimension -- 0 means no data.', count=MAXDIMS, array=True),
        _Field('d', 'sfA', help='Index value for element e of dimension d = sfA[d]*e + sfB[d].', count=MAXDIMS, array=True),
        _Field('d', 'sfB', help='Index value for element e of dimension d = sfA[d]*e + sfB[d].', count=MAXDIMS, array=True),
        # SI units
        _Field('c', 'dataUnits', default=0, help='Natural data units go here - null if none.', count=MAX_UNIT_CHARS+1, array=True),
        _Field('c', 'dimUnits', default=0, help='Natural dimension units go here - null if none.', count=(MAXDIMS, MAX_UNIT_CHARS+1), array=True),
        _Field('h', 'fsValid', help='TRUE if full scale values have meaning.'),
        _Field('h', 'whpad3', default=0, help='Reserved. Write zero. Ignore on read.'),
        _Field('d', 'topFullScale', help='The max and max full scale value for wave'), # sic, probably "max and min"
        _Field('d', 'botFullScale', help='The max and max full scale value for wave.'), # sic, probably "max and min"
        _Field('P', 'dataEUnits', default=0, help='Used in memory only. Write zero. Ignore on read.'),
        _Field('P', 'dimEUnits', default=0, help='Used in memory only. Write zero.  Ignore on read.', count=MAXDIMS, array=True),
        _Field('P', 'dimLabels', default=0, help='Used in memory only. Write zero.  Ignore on read.', count=MAXDIMS, array=True),
        _Field('P', 'waveNoteH', default=0, help='Used in memory only. Write zero. Ignore on read.'),
        _Field('l', 'whUnused', default=0, help='Reserved. Write zero. Ignore on read.', count=16, array=True),
        # The following stuff is considered private to Igor.
        _Field('h', 'aModified', default=0, help='Used in memory only. Write zero. Ignore on read.'),
        _Field('h', 'wModified', default=0, help='Used in memory only. Write zero. Ignore on read.'),
        _Field('h', 'swModified', default=0, help='Used in memory only. Write zero. Ignore on read.'),
        _Field('c', 'useBits', default=0, help='Used in memory only. Write zero. Ignore on read.'),
        _Field('c', 'kindBits', default=0, help='Reserved. Write zero. Ignore on read.'),
        _Field('P', 'formula', default=0, help='Used in memory only. Write zero. Ignore on read.'),
        _Field('l', 'depID', default=0, help='Used in memory only. Write zero. Ignore on read.'),
        _Field('h', 'whpad4', default=0, help='Reserved. Write zero. Ignore on read.'),
        _Field('h', 'srcFldr', default=0, help='Used in memory only. Write zero. Ignore on read.'),
        _Field('P', 'fileName', default=0, help='Used in memory only. Write zero. Ignore on read.'),
        _Field('P', 'sIndices', default=0, help='Used in memory only. Write zero. Ignore on read.'),
        ])


class DynamicWaveDataField1 (_DynamicField):
    def pre_pack(self, parents, data):
        raise NotImplementedError()

    def pre_unpack(self, parents, data):
        full_structure = parents[0]
        wave_structure = parents[-1]
        wave_header_structure = wave_structure.fields[1].format
        wave_data = self._get_structure_data(parents, data, wave_structure)
        version = data['version']
        bin_header = wave_data['bin_header']
        wave_header = wave_data['wave_header']

        self.count = wave_header['npnts']
        self.data_size = self._get_size(bin_header, wave_header_structure.size)

        type_ = TYPE_TABLE.get(wave_header['type'], None)
        if type_:
            self.shape = self._get_shape(bin_header, wave_header)
        else:  # text wave
            type_ = _numpy.dtype('S1')
            self.shape = (self.data_size,)
        # dtype() wrapping to avoid numpy.generic and
        # getset_descriptor issues with the builtin numpy types
        # (e.g. int32).  It has no effect on our local complex
        # integers.
        self.dtype = _numpy.dtype(type_).newbyteorder(
            wave_structure.byte_order)
        if (version == 3 and
            self.count > 0 and
            bin_header['formulaSize'] > 0 and
            self.data_size == 0):
            """From TN003:

            Igor Pro 2.00 included support for dependency formulae. If
            a wave was governed by a dependency formula then the
            actual wave data was not written to disk for that wave,
            because on loading the wave Igor could recalculate the
            data. However,this prevented the wave from being loaded
            into an experiment other than the original
            experiment. Consequently, in a version of Igor Pro 3.0x,
            we changed it so that the wave data was written even if
            the wave was governed by a dependency formula. When
            reading a binary wave file, you can detect that the wave
            file does not contain the wave data by examining the
            wfmSize, formulaSize and npnts fields. If npnts is greater
            than zero and formulaSize is greater than zero and
            the waveDataSize as calculated above is zero, then this is
            a file governed by a dependency formula that was written
            without the actual wave data.
            """
            self.shape = (0,)
        elif TYPE_TABLE.get(wave_header['type'], None) is not None:
            assert self.data_size == self.count * self.dtype.itemsize, (
                self.data_size, self.count, self.dtype.itemsize, self.dtype)
        else:
            assert self.data_size >= 0, (
                bin_header['wfmSize'], wave_header_structure.size)

    def _get_size(self, bin_header, wave_header_size):
        return bin_header['wfmSize'] - wave_header_size - 16

    def _get_shape(self, bin_header, wave_header):
        return (self.count,)

    def unpack(self, stream):
        data_b = stream.read(self.data_size)
        try:
            data = _numpy.ndarray(
                shape=self.shape,
                dtype=self.dtype,
                buffer=data_b,
                order='F',
                )
        except:
            _LOG.error(
                'could not reshape data from {} to {}'.format(
                    self.shape, data_b))
            raise
        return data


class DynamicWaveDataField5 (DynamicWaveDataField1):
    "Adds support for multidimensional data."
    def _get_size(self, bin_header, wave_header_size):
        return bin_header['wfmSize'] - wave_header_size

    def _get_shape(self, bin_header, wave_header):
        return [n for n in wave_header['nDim'] if n > 0] or (0,)


# End IGOR constants and typedefs from IgorBin.h


class DynamicStringField (StaticStringField):
    _size_field = None

    def pre_unpack(self, parents, data):
        size = self._get_size_data(parents, data)
        if self._array_size_field:
            self.counts = size
            self.count = sum(self.counts)
        else:
            self.count = size
        self.setup()

    def _get_size_data(self, parents, data):
        wave_structure = parents[-1]
        wave_data = self._get_structure_data(parents, data, wave_structure)
        bin_header = wave_data['bin_header']
        return bin_header[self._size_field]


class DynamicWaveNoteField (DynamicStringField):
    _size_field = 'noteSize'


class DynamicDependencyFormulaField (DynamicStringField):
    """Optional wave dependency formula

    Excerpted from TN003:

    A wave has a dependency formula if it has been bound by a
    statement such as "wave0 := sin(x)". In this example, the
    dependency formula is "sin(x)". The formula is stored with
    no trailing null byte.
    """
    _size_field = 'formulaSize'
    # Except when it is stored with a trailing null byte :p.  See, for
    # example, test/data/mac-version3Dependent.ibw.
    _null_terminated = True


class DynamicDataUnitsField (DynamicStringField):
    """Optional extended data units data

    Excerpted from TN003:

    dataUnits - Present in versions 1, 2, 3, 5. The dataUnits field
      stores the units for the data represented by the wave. It is a C
      string terminated with a null character. This field supports
      units of 0 to 3 bytes. In version 1, 2 and 3 files, longer units
      can not be represented. In version 5 files, longer units can be
      stored using the optional extended data units section of the
      file.
    """
    _size_field = 'dataEUnitsSize'


class DynamicDimensionUnitsField (DynamicStringField):
    """Optional extended dimension units data

    Excerpted from TN003:

    xUnits - Present in versions 1, 2, 3. The xUnits field stores the
      X units for a wave. It is a C string terminated with a null
      character.  This field supports units of 0 to 3 bytes. In
      version 1, 2 and 3 files, longer units can not be represented.

    dimUnits - Present in version 5 only. This field is an array of 4
      strings, one for each possible wave dimension. Each string
      supports units of 0 to 3 bytes. Longer units can be stored using
      the optional extended dimension units section of the file.
    """
    _size_field = 'dimEUnitsSize'
    _array_size_field = True


class DynamicLabelsField (DynamicStringField):
    """Optional dimension label data

    From TN003:

    If the wave has dimension labels for dimension d then the
    dimLabelsSize[d] field of the BinHeader5 structure will be
    non-zero.

    A wave will have dimension labels if a SetDimLabel command has
    been executed on it.

    A 3 point 1D wave has 4 dimension labels. The first dimension
    label is the label for the dimension as a whole. The next three
    dimension labels are the labels for rows 0, 1, and 2. When Igor
    writes dimension labels to disk, it writes each dimension label as
    a C string (null-terminated) in a field of 32 bytes.
    """
    _size_field = 'dimLabelsSize'
    _array_size_field = True

    def post_unpack(self, parents, data):
        wave_structure = parents[-1]
        wave_data = self._get_structure_data(parents, data, wave_structure)
        bin_header = wave_data['bin_header']
        d = wave_data[self.name]
        dim_labels = []
        start = 0
        for size in bin_header[self._size_field]:
            end = start + size
            if end > start:
                dim_data = d[start:end]
                chunks = []
                for i in range(size//32):
                    chunks.append(dim_data[32*i:32*(i+1)])
                labels = [b'']
                for chunk in chunks:
                    labels[-1] = labels[-1] + b''.join(chunk)
                    if b'\x00' in chunk:
                        labels.append(b'')
                labels.pop(-1)
                start = end
            else:
                labels = []
            dim_labels.append(labels)
        wave_data[self.name] = dim_labels


class DynamicStringIndicesDataField (_DynamicField):
    """String indices used for text waves only
    """
    def pre_pack(self, parents, data):
        raise NotImplementedError()

    def pre_unpack(self, parents, data):
        wave_structure = parents[-1]
        wave_data = self._get_structure_data(parents, data, wave_structure)
        bin_header = wave_data['bin_header']
        wave_header = wave_data['wave_header']
        self.string_indices_size = bin_header['sIndicesSize']
        self.count = self.string_indices_size // 4
        if self.count:  # make sure we're in a text wave
            assert TYPE_TABLE[wave_header['type']] is None, wave_header
        self.setup()

    def post_unpack(self, parents, data):
        if not self.count:
            return
        wave_structure = parents[-1]
        wave_data = self._get_structure_data(parents, data, wave_structure)
        wave_header = wave_data['wave_header']
        wdata = wave_data['wData']
        strings = []
        start = 0
        for i,offset in enumerate(wave_data['sIndices']):
            if offset > start:
                chars = wdata[start:offset]
                strings.append(b''.join(chars))
                start = offset
            elif offset == start:
                strings.append(b'')
            else:
                raise ValueError((offset, wave_data['sIndices']))
        wdata = _numpy.array(strings)
        shape = [n for n in wave_header['nDim'] if n > 0] or (0,)
        try:
            wdata = wdata.reshape(shape)
        except ValueError:
            _LOG.error(
                'could not reshape strings from {} to {}'.format(
                    shape, wdata.shape))
            raise
        wave_data['wData'] = wdata


class DynamicVersionField (_DynamicField):
    def pre_pack(self, parents, byte_order):
        raise NotImplementedError()

    def post_unpack(self, parents, data):
        wave_structure = parents[-1]
        wave_data = self._get_structure_data(parents, data, wave_structure)
        version = wave_data['version']
        if wave_structure.byte_order in '@=':
            need_to_reorder_bytes = _need_to_reorder_bytes(version)
            wave_structure.byte_order = _byte_order(need_to_reorder_bytes)
            _LOG.debug(
                'get byte order from version: {} (reorder? {})'.format(
                    wave_structure.byte_order, need_to_reorder_bytes))
        else:
            need_to_reorder_bytes = False

        old_format = wave_structure.fields[-1].format
        if version == 1:
            wave_structure.fields[-1].format = Wave1
        elif version == 2:
            wave_structure.fields[-1].format = Wave2
        elif version == 3:
            wave_structure.fields[-1].format = Wave3
        elif version == 5:
            wave_structure.fields[-1].format = Wave5
        elif not need_to_reorder_bytes:
            raise ValueError(
                'invalid binary wave version: {}'.format(version))

        if wave_structure.fields[-1].format != old_format:
            _LOG.debug('change wave headers from {} to {}'.format(
                    old_format, wave_structure.fields[-1].format))
            wave_structure.setup()
        elif need_to_reorder_bytes:
            wave_structure.setup()

        # we might need to unpack again with the new byte order
        return need_to_reorder_bytes


class DynamicWaveField (_DynamicField):
    def post_unpack(self, parents, data):
        return
        raise NotImplementedError()  # TODO
        checksum_size = bin.size + wave.size
        wave_structure = parents[-1]
        if version == 5:
            # Version 5 checksum does not include the wData field.
            checksum_size -= 4
        c = _checksum(b, parents[-1].byte_order, 0, checksum_size)
        if c != 0:
            raise ValueError(
                ('This does not appear to be a valid Igor binary wave file.  '
                 'Error in checksum: should be 0, is {}.').format(c))

Wave1 = _DynamicStructure(
    name='Wave1',
    fields=[
        _Field(BinHeader1, 'bin_header', help='Binary wave header'),
        _Field(WaveHeader2, 'wave_header', help='Wave header'),
        DynamicWaveDataField1('f', 'wData', help='The start of the array of waveform data.', count=0, array=True),
        ])

Wave2 = _DynamicStructure(
    name='Wave2',
    fields=[
        _Field(BinHeader2, 'bin_header', help='Binary wave header'),
        _Field(WaveHeader2, 'wave_header', help='Wave header'),
        DynamicWaveDataField1('f', 'wData', help='The start of the array of waveform data.', count=0, array=True),
        _Field('x', 'padding', help='16 bytes of padding in versions 2 and 3.', count=16, array=True),
        DynamicWaveNoteField('c', 'note', help='Optional wave note data', count=0, array=True),
        ])

Wave3 = _DynamicStructure(
    name='Wave3',
    fields=[
        _Field(BinHeader3, 'bin_header', help='Binary wave header'),
        _Field(WaveHeader2, 'wave_header', help='Wave header'),
        DynamicWaveDataField1('f', 'wData', help='The start of the array of waveform data.', count=0, array=True),
        _Field('x', 'padding', help='16 bytes of padding in versions 2 and 3.', count=16, array=True),
        DynamicWaveNoteField('c', 'note', help='Optional wave note data', count=0, array=True),
        DynamicDependencyFormulaField('c', 'formula', help='Optional wave dependency formula', count=0, array=True),
        ])

Wave5 = _DynamicStructure(
    name='Wave5',
    fields=[
        _Field(BinHeader5, 'bin_header', help='Binary wave header'),
        _Field(WaveHeader5, 'wave_header', help='Wave header'),
        DynamicWaveDataField5('f', 'wData', help='The start of the array of waveform data.', count=0, array=True),
        DynamicDependencyFormulaField('c', 'formula', help='Optional wave dependency formula.', count=0, array=True),
        DynamicWaveNoteField('c', 'note', help='Optional wave note data.', count=0, array=True),
        DynamicDataUnitsField('c', 'data_units', help='Optional extended data units data.', count=0, array=True),
        DynamicDimensionUnitsField('c', 'dimension_units', help='Optional dimension label data', count=0, array=True),
        DynamicLabelsField('c', 'labels', help="Optional dimension label data", count=0, array=True),
        DynamicStringIndicesDataField('P', 'sIndices', help='Dynamic string indices for text waves.', count=0, array=True),
        ])

Wave = _DynamicStructure(
    name='Wave',
    fields=[
        DynamicVersionField('h', 'version', help='Version number for backwards compatibility.'),
        DynamicWaveField(Wave1, 'wave', help='The rest of the wave data.'),
        ])


def load(filename):
    if hasattr(filename, 'read'):
        f = filename  # filename is actually a stream object
    else:
        f = open(filename, 'rb')
    try:
        Wave.byte_order = '='
        Wave.setup()
        data = Wave.unpack_stream(f)
    finally:
        if not hasattr(filename, 'read'):
            f.close()