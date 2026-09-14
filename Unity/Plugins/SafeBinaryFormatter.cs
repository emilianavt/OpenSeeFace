using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace OpenSee {

public class SafeSerializationException : Exception {
    public SafeSerializationException(string message) : base(message) {}
}

internal static class NrbfRecord {
    public const byte SerializedStreamHeader = 0;
    public const byte ClassWithId = 1;
    public const byte SystemClassWithMembersAndTypes = 4;
    public const byte ClassWithMembersAndTypes = 5;
    public const byte BinaryObjectString = 6;
    public const byte BinaryArray = 7;
    public const byte MemberReference = 9;
    public const byte ObjectNull = 10;
    public const byte MessageEnd = 11;
    public const byte BinaryLibrary = 12;
    public const byte ObjectNullMultiple256 = 13;
    public const byte ObjectNullMultiple = 14;
    public const byte ArraySinglePrimitive = 15;
    public const byte ArraySingleObject = 16;
    public const byte ArraySingleString = 17;
}

internal static class NrbfBinaryType {
    public const byte Primitive = 0;
    public const byte String = 1;
    public const byte Object = 2;
    public const byte SystemClass = 3;
    public const byte Class = 4;
    public const byte ObjectArray = 5;
    public const byte StringArray = 6;
    public const byte PrimitiveArray = 7;
}

public static class NrbfPrimitiveType {
    public const byte Boolean = 1;
    public const byte Byte = 2;
    public const byte Double = 6;
    public const byte Int16 = 7;
    public const byte Int32 = 8;
    public const byte Int64 = 9;
    public const byte Single = 11;
    public const byte String = 18;
}

public class SafeObject {
    public string ClassName;
    public readonly Dictionary<string, object> Members = new Dictionary<string, object>();

    public bool TryGet(string name, out object value) {
        return Members.TryGetValue(name, out value);
    }

    private object Get(string name) {
        object value;
        if (!Members.TryGetValue(name, out value))
            return null;
        return value;
    }

    private T GetAs<T>(string name, T defaultValue) {
        object value = Get(name);
        if (value == null)
            return defaultValue;
        if (!(value is T))
            throw new SafeSerializationException("Member " + name + " of " + ClassName + " has type " + value.GetType().Name + ", expected " + typeof(T).Name);
        return (T)value;
    }

    public bool GetBoolean(string name, bool defaultValue) { return GetAs<bool>(name, defaultValue); }
    public int GetInt32(string name, int defaultValue) { return GetAs<int>(name, defaultValue); }
    public string GetString(string name) { return GetAs<string>(name, null); }
    public byte[] GetByteArray(string name) { return GetAs<byte[]>(name, null); }
    public int[] GetInt32Array(string name) { return GetAs<int[]>(name, null); }
    public float[] GetSingleArray(string name) { return GetAs<float[]>(name, null); }
    public double[] GetDoubleArray(string name) { return GetAs<double[]>(name, null); }
    public string[] GetStringArray(string name) {
        object[] array = GetAs<object[]>(name, null);
        if (array == null)
            return null;
        string[] result = new string[array.Length];
        for (int i = 0; i < array.Length; i++) {
            if (array[i] != null && !(array[i] is string))
                throw new SafeSerializationException("Member " + name + " of " + ClassName + " is not a string array");
            result[i] = (string)array[i];
        }
        return result;
    }
    public object[] GetObjectArray(string name) { return GetAs<object[]>(name, null); }
    public SafeObject GetObject(string name) { return GetAs<SafeObject>(name, null); }
}

public class SafeBinaryReader {
    private BinaryReader reader;
    private long streamLength;

    private class ClassMetadata {
        public string name;
        public string[] memberNames;
        public byte[] binaryTypes;
        public object[] additionalInfos;
    }
    private Dictionary<int, ClassMetadata> classMetadata = new Dictionary<int, ClassMetadata>();

    private Dictionary<int, object> objects = new Dictionary<int, object>();

    private struct Fixup {
        public SafeObject targetObject;
        public string memberName;
        public object[] targetArray;
        public int index;
        public int objectId;
    }
    private List<Fixup> fixups = new List<Fixup>();

    private int depth = 0;
    private const int maxDepth = 64;

    private SafeBinaryReader(Stream stream) {
        reader = new BinaryReader(stream, Encoding.UTF8);
        streamLength = stream.Length;
    }

    public static object Deserialize(Stream stream) {
        return new SafeBinaryReader(stream).Parse();
    }

    public static object Deserialize(byte[] data) {
        using (MemoryStream stream = new MemoryStream(data, false))
            return Deserialize(stream);
    }

    private SafeSerializationException Error(string message) {
        return new SafeSerializationException(message + " (at " + reader.BaseStream.Position + ")");
    }

    private object Parse() {
        if (reader.ReadByte() != NrbfRecord.SerializedStreamHeader)
            throw Error("Invalid header");
        int rootId = reader.ReadInt32();
        reader.ReadInt32();
        int major = reader.ReadInt32();
        int minor = reader.ReadInt32();
        if (major != 1 || minor != 0)
            throw Error("Unsupported serialization format version " + major + "." + minor);

        while (true) {
            byte recordType = reader.ReadByte();
            if (recordType == NrbfRecord.MessageEnd)
                break;
            ParseRecord(recordType);
        }

        foreach (Fixup fixup in fixups) {
            object value;
            if (!objects.TryGetValue(fixup.objectId, out value))
                throw new SafeSerializationException("Unresolved member reference to object " + fixup.objectId);
            if (fixup.targetObject != null)
                fixup.targetObject.Members[fixup.memberName] = value;
            else
                fixup.targetArray[fixup.index] = value;
        }

        object root;
        if (!objects.TryGetValue(rootId, out root))
            throw new SafeSerializationException("Root object " + rootId + " missing from stream");
        return root;
    }

    private void RegisterObject(int objectId, object value) {
        if (objects.ContainsKey(objectId))
            throw Error("Duplicate object id " + objectId);
        objects[objectId] = value;
    }

    private object ParseRecord(byte recordType) {
        if (++depth > maxDepth)
            throw Error("Nesting too deep");
        try {
            switch (recordType) {
                case NrbfRecord.BinaryLibrary: {
                    reader.ReadInt32();
                    ReadString();
                    return ParseRecord(reader.ReadByte());
                }
                case NrbfRecord.ClassWithMembersAndTypes:
                case NrbfRecord.SystemClassWithMembersAndTypes:
                    return ParseClass(recordType == NrbfRecord.ClassWithMembersAndTypes);
                case NrbfRecord.ClassWithId: {
                    int objectId = reader.ReadInt32();
                    int metadataId = reader.ReadInt32();
                    ClassMetadata metadata;
                    if (!classMetadata.TryGetValue(metadataId, out metadata))
                        throw Error("ClassWithId refers to unknown metadata " + metadataId);
                    return ParseClassMembers(objectId, metadata);
                }
                case NrbfRecord.BinaryObjectString: {
                    int objectId = reader.ReadInt32();
                    string value = ReadString();
                    RegisterObject(objectId, value);
                    return value;
                }
                case NrbfRecord.ArraySinglePrimitive:
                    return ParsePrimitiveArray();
                case NrbfRecord.ArraySingleString:
                case NrbfRecord.ArraySingleObject: {
                    int objectId = reader.ReadInt32();
                    int length = ReadArrayLength(1);
                    object[] array = new object[length];
                    RegisterObject(objectId, array);
                    ParseArrayItems(array, length);
                    return array;
                }
                case NrbfRecord.BinaryArray:
                    return ParseBinaryArray();
                default:
                    throw Error("Unsupported record type " + recordType);
            }
        } finally {
            depth--;
        }
    }

    private object ParseClass(bool hasLibraryId) {
        int objectId = reader.ReadInt32();
        ClassMetadata metadata = new ClassMetadata();
        metadata.name = ReadString();
        int memberCount = reader.ReadInt32();
        if (memberCount < 0 || memberCount > 1024)
            throw Error("Implausible member count " + memberCount);
        metadata.memberNames = new string[memberCount];
        for (int i = 0; i < memberCount; i++)
            metadata.memberNames[i] = ReadString();
        metadata.binaryTypes = new byte[memberCount];
        for (int i = 0; i < memberCount; i++)
            metadata.binaryTypes[i] = reader.ReadByte();
        metadata.additionalInfos = new object[memberCount];
        for (int i = 0; i < memberCount; i++)
            metadata.additionalInfos[i] = ReadTypeAdditionalInfo(metadata.binaryTypes[i]);
        if (hasLibraryId)
            reader.ReadInt32();
        classMetadata[objectId] = metadata;
        return ParseClassMembers(objectId, metadata);
    }

    private SafeObject ParseClassMembers(int objectId, ClassMetadata metadata) {
        SafeObject obj = new SafeObject();
        obj.ClassName = metadata.name;
        RegisterObject(objectId, obj);
        for (int i = 0; i < metadata.memberNames.Length; i++) {
            string name = metadata.memberNames[i];
            byte binaryType = metadata.binaryTypes[i];
            if (binaryType == NrbfBinaryType.Primitive) {
                obj.Members[name] = ReadPrimitiveValue((byte)metadata.additionalInfos[i]);
            } else {
                int referenceId;
                object value = ParseMemberRecord(out referenceId);
                if (referenceId != 0) {
                    Fixup fixup = new Fixup();
                    fixup.targetObject = obj;
                    fixup.memberName = name;
                    fixup.objectId = referenceId;
                    fixups.Add(fixup);
                } else {
                    obj.Members[name] = value;
                }
            }
        }
        return obj;
    }

    private object ParseMemberRecord(out int referenceId) {
        referenceId = 0;
        byte recordType = reader.ReadByte();
        switch (recordType) {
            case NrbfRecord.MemberReference: {
                int id = reader.ReadInt32();
                if (id == 0)
                    throw Error("Member reference to object id 0");
                referenceId = id;
                return null;
            }
            case NrbfRecord.ObjectNull:
                return null;
            default:
                return ParseRecord(recordType);
        }
    }

    private object ParsePrimitiveArray() {
        int objectId = reader.ReadInt32();
        int length = ReadArrayLength(1);
        byte primitiveType = reader.ReadByte();
        object array;
        switch (primitiveType) {
            case NrbfPrimitiveType.Byte: {
                CheckAvailable((long)length);
                byte[] bytes = reader.ReadBytes(length);
                if (bytes.Length != length)
                    throw Error("Truncated byte array");
                array = bytes;
                break;
            }
            case NrbfPrimitiveType.Int32: {
                CheckAvailable((long)length * 4);
                int[] values = new int[length];
                for (int i = 0; i < length; i++)
                    values[i] = reader.ReadInt32();
                array = values;
                break;
            }
            case NrbfPrimitiveType.Single: {
                CheckAvailable((long)length * 4);
                float[] values = new float[length];
                for (int i = 0; i < length; i++)
                    values[i] = reader.ReadSingle();
                array = values;
                break;
            }
            case NrbfPrimitiveType.Double: {
                CheckAvailable((long)length * 8);
                double[] values = new double[length];
                for (int i = 0; i < length; i++)
                    values[i] = reader.ReadDouble();
                array = values;
                break;
            }
            default:
                throw Error("Unsupported primitive array type " + primitiveType);
        }
        RegisterObject(objectId, array);
        return array;
    }

    private object ParseBinaryArray() {
        int objectId = reader.ReadInt32();
        byte arrayType = reader.ReadByte();
        if (arrayType != 0 && arrayType != 1)
            throw Error("Unsupported binary array type " + arrayType);
        int rank = reader.ReadInt32();
        if (rank != 1)
            throw Error("Unsupported array rank " + rank);
        int length = ReadArrayLength(1);
        byte binaryType = reader.ReadByte();
        object additionalInfo = ReadTypeAdditionalInfo(binaryType);
        if (binaryType == NrbfBinaryType.Primitive) {
            byte primitiveType = (byte)additionalInfo;
            switch (primitiveType) {
                case NrbfPrimitiveType.Single: {
                    CheckAvailable((long)length * 4);
                    float[] values = new float[length];
                    for (int i = 0; i < length; i++)
                        values[i] = reader.ReadSingle();
                    RegisterObject(objectId, values);
                    return values;
                }
                default:
                    throw Error("Unsupported binary array primitive type " + primitiveType);
            }
        }
        object[] array = new object[length];
        RegisterObject(objectId, array);
        ParseArrayItems(array, length);
        return array;
    }

    private void ParseArrayItems(object[] array, int length) {
        int i = 0;
        while (i < length) {
            byte recordType = reader.ReadByte();
            switch (recordType) {
                case NrbfRecord.ObjectNull:
                    array[i++] = null;
                    break;
                case NrbfRecord.ObjectNullMultiple256: {
                    int count = reader.ReadByte();
                    if (i + count > length)
                        throw Error("Null run exceeds array length");
                    i += count;
                    break;
                }
                case NrbfRecord.ObjectNullMultiple: {
                    int count = reader.ReadInt32();
                    if (count < 0 || i + count > length)
                        throw Error("Null run exceeds array length");
                    i += count;
                    break;
                }
                case NrbfRecord.MemberReference: {
                    int id = reader.ReadInt32();
                    if (id == 0)
                        throw Error("Member reference to object id 0");
                    Fixup fixup = new Fixup();
                    fixup.targetArray = array;
                    fixup.index = i;
                    fixup.objectId = id;
                    fixups.Add(fixup);
                    i++;
                    break;
                }
                default:
                    array[i++] = ParseRecord(recordType);
                    break;
            }
        }
    }

    private object ReadTypeAdditionalInfo(byte binaryType) {
        switch (binaryType) {
            case NrbfBinaryType.Primitive:
            case NrbfBinaryType.PrimitiveArray:
                return reader.ReadByte();
            case NrbfBinaryType.String:
            case NrbfBinaryType.Object:
            case NrbfBinaryType.ObjectArray:
            case NrbfBinaryType.StringArray:
                return null;
            case NrbfBinaryType.SystemClass:
                return ReadString();
            case NrbfBinaryType.Class: {
                string name = ReadString();
                reader.ReadInt32();
                return name;
            }
            default:
                throw Error("Unsupported binary type " + binaryType);
        }
    }

    private object ReadPrimitiveValue(byte primitiveType) {
        switch (primitiveType) {
            case NrbfPrimitiveType.Boolean: return reader.ReadBoolean();
            case NrbfPrimitiveType.Byte: return reader.ReadByte();
            case NrbfPrimitiveType.Double: return reader.ReadDouble();
            case NrbfPrimitiveType.Int16: return reader.ReadInt16();
            case NrbfPrimitiveType.Int32: return reader.ReadInt32();
            case NrbfPrimitiveType.Int64: return reader.ReadInt64();
            case NrbfPrimitiveType.Single: return reader.ReadSingle();
            case NrbfPrimitiveType.String: return ReadString();
            default:
                throw Error("Unsupported primitive type " + primitiveType);
        }
    }

    private string ReadString() {
        long position = reader.BaseStream.Position;
        int length = 0;
        int shift = 0;
        while (true) {
            byte b = reader.ReadByte();
            length |= (b & 0x7f) << shift;
            if ((b & 0x80) == 0)
                break;
            shift += 7;
            if (shift > 28)
                throw Error("Invalid string length prefix");
        }
        if (length < 0)
            throw Error("Invalid string length " + length);
        CheckAvailable(length);
        reader.BaseStream.Position = position;
        return reader.ReadString();
    }

    private int ReadArrayLength(int minimumElementSize) {
        int length = reader.ReadInt32();
        if (length < 0)
            throw Error("Negative array length");
        CheckAvailable((long)length * minimumElementSize);
        return length;
    }

    private void CheckAvailable(long byteCount) {
        if (byteCount > streamLength - reader.BaseStream.Position)
            throw Error("Data of size " + byteCount + " exceeds remaining stream size");
    }
}


public struct SafeMemberType {
    public byte binaryType;
    public byte primitiveType;
    public string className;
    public int libraryId;

    public static SafeMemberType Primitive(byte primitiveType) {
        SafeMemberType t = new SafeMemberType();
        t.binaryType = NrbfBinaryType.Primitive;
        t.primitiveType = primitiveType;
        return t;
    }

    public static SafeMemberType PrimitiveArray(byte primitiveType) {
        SafeMemberType t = new SafeMemberType();
        t.binaryType = NrbfBinaryType.PrimitiveArray;
        t.primitiveType = primitiveType;
        return t;
    }

    public static SafeMemberType String() {
        SafeMemberType t = new SafeMemberType();
        t.binaryType = NrbfBinaryType.String;
        return t;
    }

    public static SafeMemberType StringArray() {
        SafeMemberType t = new SafeMemberType();
        t.binaryType = NrbfBinaryType.StringArray;
        return t;
    }

    public static SafeMemberType SystemClass(string className) {
        SafeMemberType t = new SafeMemberType();
        t.binaryType = NrbfBinaryType.SystemClass;
        t.className = className;
        return t;
    }

    public static SafeMemberType Class(string className, int libraryId) {
        SafeMemberType t = new SafeMemberType();
        t.binaryType = NrbfBinaryType.Class;
        t.className = className;
        t.libraryId = libraryId;
        return t;
    }
}

public class SafeBinaryWriter {
    private BinaryWriter writer;
    private int nextId = 1;

    public SafeBinaryWriter(Stream stream) {
        writer = new BinaryWriter(stream, Encoding.UTF8);
    }

    public int AllocateId() {
        return nextId++;
    }

    public void Flush() {
        writer.Flush();
    }

    public void WriteHeader(int rootId) {
        writer.Write(NrbfRecord.SerializedStreamHeader);
        writer.Write(rootId);
        writer.Write(-1);
        writer.Write(1);
        writer.Write(0);
    }

    public void WriteMessageEnd() {
        writer.Write(NrbfRecord.MessageEnd);
        writer.Flush();
    }

    public void WriteBinaryLibrary(int libraryId, string libraryName) {
        writer.Write(NrbfRecord.BinaryLibrary);
        writer.Write(libraryId);
        writer.Write(libraryName);
    }

    public void WriteClassHeader(int objectId, string className, string[] memberNames, SafeMemberType[] memberTypes, int libraryId) {
        writer.Write(libraryId >= 0 ? NrbfRecord.ClassWithMembersAndTypes : NrbfRecord.SystemClassWithMembersAndTypes);
        writer.Write(objectId);
        writer.Write(className);
        writer.Write(memberNames.Length);
        foreach (string name in memberNames)
            writer.Write(name);
        foreach (SafeMemberType type in memberTypes)
            writer.Write(type.binaryType);
        foreach (SafeMemberType type in memberTypes) {
            switch (type.binaryType) {
                case NrbfBinaryType.Primitive:
                case NrbfBinaryType.PrimitiveArray:
                    writer.Write(type.primitiveType);
                    break;
                case NrbfBinaryType.SystemClass:
                    writer.Write(type.className);
                    break;
                case NrbfBinaryType.Class:
                    writer.Write(type.className);
                    writer.Write(type.libraryId);
                    break;
            }
        }
        if (libraryId >= 0)
            writer.Write(libraryId);
    }

    public void WriteClassWithId(int objectId, int metadataId) {
        writer.Write(NrbfRecord.ClassWithId);
        writer.Write(objectId);
        writer.Write(metadataId);
    }

    public void WriteMemberReference(int objectId) {
        writer.Write(NrbfRecord.MemberReference);
        writer.Write(objectId);
    }

    public void WriteObjectNull() {
        writer.Write(NrbfRecord.ObjectNull);
    }

    public void WriteBinaryObjectString(int objectId, string value) {
        writer.Write(NrbfRecord.BinaryObjectString);
        writer.Write(objectId);
        writer.Write(value);
    }

    public void WriteValue(bool value) { writer.Write(value); }
    public void WriteValue(int value) { writer.Write(value); }
    public void WriteValue(float value) { writer.Write(value); }
    public void WriteValue(double value) { writer.Write(value); }

    public void WriteArraySinglePrimitive(int objectId, byte[] values) {
        WritePrimitiveArrayHeader(objectId, values.Length, NrbfPrimitiveType.Byte);
        writer.Write(values);
    }

    public void WriteArraySinglePrimitive(int objectId, int[] values) {
        WritePrimitiveArrayHeader(objectId, values.Length, NrbfPrimitiveType.Int32);
        foreach (int value in values)
            writer.Write(value);
    }

    public void WriteArraySinglePrimitive(int objectId, float[] values) {
        WritePrimitiveArrayHeader(objectId, values.Length, NrbfPrimitiveType.Single);
        foreach (float value in values)
            writer.Write(value);
    }

    public void WriteArraySinglePrimitive(int objectId, double[] values) {
        WritePrimitiveArrayHeader(objectId, values.Length, NrbfPrimitiveType.Double);
        foreach (double value in values)
            writer.Write(value);
    }

    private void WritePrimitiveArrayHeader(int objectId, int length, byte primitiveType) {
        writer.Write(NrbfRecord.ArraySinglePrimitive);
        writer.Write(objectId);
        writer.Write(length);
        writer.Write(primitiveType);
    }

    public void WriteStringArrayHeader(int objectId, int length) {
        writer.Write(NrbfRecord.ArraySingleString);
        writer.Write(objectId);
        writer.Write(length);
    }

    public void WriteBinaryArrayHeader(int objectId, byte arrayType, int length, SafeMemberType elementType) {
        writer.Write(NrbfRecord.BinaryArray);
        writer.Write(objectId);
        writer.Write(arrayType);
        writer.Write(1);
        writer.Write(length);
        writer.Write(elementType.binaryType);
        switch (elementType.binaryType) {
            case NrbfBinaryType.Primitive:
            case NrbfBinaryType.PrimitiveArray:
                writer.Write(elementType.primitiveType);
                break;
            case NrbfBinaryType.SystemClass:
                writer.Write(elementType.className);
                break;
            case NrbfBinaryType.Class:
                writer.Write(elementType.className);
                writer.Write(elementType.libraryId);
                break;
        }
    }
}

public static class SafeCollections {
    private const string mscorlib = "mscorlib, Version=4.0.0.0, Culture=neutral, PublicKeyToken=b77a5c561934e089";
    private const string singleListName = "System.Collections.Generic.List`1[[System.Single[], " + mscorlib + "]]";
    public const string DictionaryClassName = "System.Collections.Generic.Dictionary`2[[System.String, " + mscorlib + "],[" + singleListName + ", " + mscorlib + "]]";
    private const string keyValuePairName = "System.Collections.Generic.KeyValuePair`2[[System.String, " + mscorlib + "],[" + singleListName + ", " + mscorlib + "]]";
    private const string comparerName = "System.Collections.Generic.GenericEqualityComparer`1[[System.String, " + mscorlib + "]]";
    private const string singleJaggedArrayName = "System.Single[][]";

    private static readonly int[] primes = new int[] {3, 7, 11, 17, 23, 29, 37, 47, 59, 71, 89, 107, 131, 163, 197, 239, 293, 353, 431, 521, 631, 761, 919, 1103, 1327, 1597, 1931, 2333, 2801, 3371, 4049, 4861, 5839, 7013, 8419, 10103, 12143, 14591, 17519, 21023, 25229, 30293, 36353, 43627, 52361, 62851, 75431, 90523, 108631, 130363, 156437, 187751, 225307, 270371, 324449, 389357, 467237, 560689, 672827, 807403, 968897, 1162687, 1395263, 1674319, 2009191, 2411033, 2893249};

    private static int HashSize(int count) {
        if (count == 0)
            return 0;
        foreach (int prime in primes)
            if (prime >= count)
                return prime;
        return count | 1;
    }

    public static Dictionary<string, List<float[]>> ToDictionary(SafeObject dict) {
        Dictionary<string, List<float[]>> result = new Dictionary<string, List<float[]>>();
        if (dict == null)
            return result;
        object[] pairs = dict.GetObjectArray("KeyValuePairs");
        if (pairs == null)
            return result;
        foreach (object pairObject in pairs) {
            SafeObject pair = pairObject as SafeObject;
            if (pair == null)
                throw new SafeSerializationException("Dictionary entry is not a key value pair");
            string key = pair.GetString("key");
            SafeObject list = pair.GetObject("value");
            if (key == null || list == null)
                throw new SafeSerializationException("Dictionary entry is missing key or value");
            object[] items = list.GetObjectArray("_items");
            int size = list.GetInt32("_size", items == null ? 0 : items.Length);
            if (items == null || size < 0 || size > items.Length)
                throw new SafeSerializationException("Invalid list in dictionary entry");
            List<float[]> rows = new List<float[]>(size);
            for (int i = 0; i < size; i++) {
                float[] row = items[i] as float[];
                if (row == null)
                    throw new SafeSerializationException("List item is not a float array");
                rows.Add(row);
            }
            result[key] = rows;
        }
        return result;
    }

    public static void WriteDictionary(SafeBinaryWriter writer, int objectId, Dictionary<string, List<float[]>> dict) {
        int count = dict.Count;
        int comparerId = writer.AllocateId();
        int pairArrayId = count > 0 ? writer.AllocateId() : 0;

        writer.WriteClassHeader(objectId, DictionaryClassName,
            new string[] {"Version", "Comparer", "HashSize", "KeyValuePairs"},
            new SafeMemberType[] {
                SafeMemberType.Primitive(NrbfPrimitiveType.Int32),
                SafeMemberType.SystemClass(comparerName),
                SafeMemberType.Primitive(NrbfPrimitiveType.Int32),
                SafeMemberType.SystemClass(keyValuePairName + "[]")
            }, -1);
        writer.WriteValue(count);
        writer.WriteMemberReference(comparerId);
        writer.WriteValue(HashSize(count));
        if (count > 0)
            writer.WriteMemberReference(pairArrayId);
        else
            writer.WriteObjectNull();

        writer.WriteClassHeader(comparerId, comparerName, new string[0], new SafeMemberType[0], -1);
        if (count == 0)
            return;

        int[] listIds = new int[count];
        int i = 0;
        int pairMetadataId = 0;
        writer.WriteBinaryArrayHeader(pairArrayId, 0, count, SafeMemberType.SystemClass(keyValuePairName));
        foreach (KeyValuePair<string, List<float[]>> entry in dict) {
            listIds[i] = writer.AllocateId();
            if (i == 0) {
                pairMetadataId = writer.AllocateId();
                writer.WriteClassHeader(pairMetadataId, keyValuePairName,
                    new string[] {"key", "value"},
                    new SafeMemberType[] {
                        SafeMemberType.String(),
                        SafeMemberType.SystemClass(singleListName)
                    }, -1);
            } else {
                writer.WriteClassWithId(writer.AllocateId(), pairMetadataId);
            }
            writer.WriteBinaryObjectString(writer.AllocateId(), entry.Key);
            writer.WriteMemberReference(listIds[i]);
            i++;
        }

        int listMetadataId = 0;
        i = 0;
        foreach (KeyValuePair<string, List<float[]>> entry in dict) {
            List<float[]> rows = entry.Value;
            int itemsId = writer.AllocateId();
            if (i == 0) {
                listMetadataId = listIds[i];
                writer.WriteClassHeader(listIds[i], singleListName,
                    new string[] {"_items", "_size", "_version"},
                    new SafeMemberType[] {
                        SafeMemberType.SystemClass(singleJaggedArrayName),
                        SafeMemberType.Primitive(NrbfPrimitiveType.Int32),
                        SafeMemberType.Primitive(NrbfPrimitiveType.Int32)
                    }, -1);
            } else {
                writer.WriteClassWithId(listIds[i], listMetadataId);
            }
            writer.WriteMemberReference(itemsId);
            writer.WriteValue(rows.Count);
            writer.WriteValue(rows.Count);

            int[] rowIds = new int[rows.Count];
            writer.WriteBinaryArrayHeader(itemsId, 1, rows.Count, SafeMemberType.PrimitiveArray(NrbfPrimitiveType.Single));
            for (int j = 0; j < rows.Count; j++) {
                rowIds[j] = writer.AllocateId();
                writer.WriteMemberReference(rowIds[j]);
            }
            for (int j = 0; j < rows.Count; j++)
                writer.WriteArraySinglePrimitive(rowIds[j], rows[j]);
            i++;
        }
    }
}
}
