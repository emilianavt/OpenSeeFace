using System;
using System.ComponentModel;
using System.Diagnostics;
using System.Runtime.InteropServices;

namespace OpenSee {

// Based on https://github.com/josh4364/IL2cppStartProcess/blob/master/StartExternalProcess.cs

public class OpenSeeProcessInterface {
    #region DllImport
    [DllImport("kernel32.dll", SetLastError=true, CharSet=CharSet.Auto)]
    private static extern IntPtr CreateFile(
        string lpFileName,
        EFileAccess dwDesiredAccess,
        EFileShare dwShareMode,
        SECURITY_ATTRIBUTES lpSecurityAttributes,
        ECreationDisposition dwCreationDisposition,
        EFileAttributes dwFlagsAndAttributes,
        IntPtr hTemplateFile
    );
    
    [StructLayout(LayoutKind.Sequential)]
    private struct SECURITY_ATTRIBUTES
    {
        public int nLength;
        public IntPtr lpSecurityDescriptor;
        public int bInheritHandle;
    }

    [Flags]
    enum EFileAccess : uint {
        //
        // Standart Section
        //

        AccessSystemSecurity = 0x1000000,   // AccessSystemAcl access type
        MaximumAllowed = 0x2000000,     // MaximumAllowed access type

        Delete = 0x10000,
        ReadControl = 0x20000,
        WriteDAC = 0x40000,
        WriteOwner = 0x80000,
        Synchronize = 0x100000,

        StandardRightsRequired = 0xF0000,
        StandardRightsRead = ReadControl,
        StandardRightsWrite = ReadControl,
        StandardRightsExecute = ReadControl,
        StandardRightsAll = 0x1F0000,
        SpecificRightsAll = 0xFFFF,

        FILE_READ_DATA = 0x0001,        // file & pipe
        FILE_LIST_DIRECTORY = 0x0001,       // directory
        FILE_WRITE_DATA = 0x0002,       // file & pipe
        FILE_ADD_FILE = 0x0002,         // directory
        FILE_APPEND_DATA = 0x0004,      // file
        FILE_ADD_SUBDIRECTORY = 0x0004,     // directory
        FILE_CREATE_PIPE_INSTANCE = 0x0004, // named pipe
        FILE_READ_EA = 0x0008,          // file & directory
        FILE_WRITE_EA = 0x0010,         // file & directory
        FILE_EXECUTE = 0x0020,          // file
        FILE_TRAVERSE = 0x0020,         // directory
        FILE_DELETE_CHILD = 0x0040,     // directory
        FILE_READ_ATTRIBUTES = 0x0080,      // all
        FILE_WRITE_ATTRIBUTES = 0x0100,     // all

        //
        // Generic Section
        //

        GenericRead = 0x80000000,
        GenericWrite = 0x40000000,
        GenericExecute = 0x20000000,
        GenericAll = 0x10000000,

        SPECIFIC_RIGHTS_ALL = 0x00FFFF,
        FILE_ALL_ACCESS =
            StandardRightsRequired |
            Synchronize |
            0x1FF,

        FILE_GENERIC_READ =
            StandardRightsRead |
            FILE_READ_DATA |
            FILE_READ_ATTRIBUTES |
            FILE_READ_EA |
            Synchronize,

        FILE_GENERIC_WRITE =
            StandardRightsWrite |
            FILE_WRITE_DATA |
            FILE_WRITE_ATTRIBUTES |
            FILE_WRITE_EA |
            FILE_APPEND_DATA |
            Synchronize,

        FILE_GENERIC_EXECUTE =
            StandardRightsExecute |
            FILE_READ_ATTRIBUTES |
            FILE_EXECUTE |
            Synchronize
    }

    [Flags]
    public enum EFileShare : uint
    {
        /// <summary>
        ///
        /// </summary>
        None = 0x00000000,
        /// <summary>
        /// Enables subsequent open operations on an object to request read access.
        /// Otherwise, other processes cannot open the object if they request read access.
        /// If this flag is not specified, but the object has been opened for read access, the function fails.
        /// </summary>
        Read = 0x00000001,
        /// <summary>
        /// Enables subsequent open operations on an object to request write access.
        /// Otherwise, other processes cannot open the object if they request write access.
        /// If this flag is not specified, but the object has been opened for write access, the function fails.
        /// </summary>
        Write = 0x00000002,
        /// <summary>
        /// Enables subsequent open operations on an object to request delete access.
        /// Otherwise, other processes cannot open the object if they request delete access.
        /// If this flag is not specified, but the object has been opened for delete access, the function fails.
        /// </summary>
        Delete = 0x00000004
    }

    public enum ECreationDisposition : uint
    {
        /// <summary>
        /// Creates a new file. The function fails if a specified file exists.
        /// </summary>
        New = 1,
        /// <summary>
        /// Creates a new file, always.
        /// If a file exists, the function overwrites the file, clears the existing attributes, combines the specified file attributes,
        /// and flags with FILE_ATTRIBUTE_ARCHIVE, but does not set the security descriptor that the SECURITY_ATTRIBUTES structure specifies.
        /// </summary>
        CreateAlways = 2,
        /// <summary>
        /// Opens a file. The function fails if the file does not exist.
        /// </summary>
        OpenExisting = 3,
        /// <summary>
        /// Opens a file, always.
        /// If a file does not exist, the function creates a file as if dwCreationDisposition is CREATE_NEW.
        /// </summary>
        OpenAlways = 4,
        /// <summary>
        /// Opens a file and truncates it so that its size is 0 (zero) bytes. The function fails if the file does not exist.
        /// The calling process must open the file with the GENERIC_WRITE access right.
        /// </summary>
        TruncateExisting = 5
    }

    [Flags]
    public enum EFileAttributes : uint
    {
        Readonly         = 0x00000001,
        Hidden           = 0x00000002,
        System           = 0x00000004,
        Directory        = 0x00000010,
        Archive          = 0x00000020,
        Device           = 0x00000040,
        Normal           = 0x00000080,
        Temporary        = 0x00000100,
        SparseFile       = 0x00000200,
        ReparsePoint     = 0x00000400,
        Compressed       = 0x00000800,
        Offline          = 0x00001000,
        NotContentIndexed= 0x00002000,
        Encrypted        = 0x00004000,
        Write_Through    = 0x80000000,
        Overlapped       = 0x40000000,
        NoBuffering      = 0x20000000,
        RandomAccess     = 0x10000000,
        SequentialScan   = 0x08000000,
        DeleteOnClose    = 0x04000000,
        BackupSemantics  = 0x02000000,
        PosixSemantics   = 0x01000000,
        OpenReparsePoint = 0x00200000,
        OpenNoRecall     = 0x00100000,
        FirstPipeInstance= 0x00080000
    }

    
    [DllImport("kernel32.dll", SetLastError=true)]
    private static extern UInt32 WaitForSingleObject(IntPtr hHandle, UInt32 dwMilliseconds);
    const UInt32 INFINITE = 0xFFFFFFFF;
    const UInt32 WAIT_ABANDONED = 0x00000080;
    const UInt32 WAIT_OBJECT_0 = 0x00000000;
    const UInt32 WAIT_TIMEOUT = 0x00000102;
    
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool CreateProcessW(
        string lpApplicationName,
        [In] string lpCommandLine,
        IntPtr procSecAttrs,
        IntPtr threadSecAttrs,
        bool bInheritHandles,
        ProcessCreationFlags dwCreationFlags,
        IntPtr lpEnvironment,
        string lpCurrentDirectory,
        ref STARTUPINFO lpStartupInfo,
        ref PROCESS_INFORMATION lpProcessInformation
    );

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    public static extern bool CloseHandle(IntPtr hObject);

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    public static extern bool TerminateProcess(IntPtr processHandle, uint exitCode);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern IntPtr OpenProcess(ProcessAccessRights access, bool inherit, uint processId);

    [Flags]
    private enum ProcessAccessRights : uint
    {
        PROCESS_CREATE_PROCESS = 0x0080, //  Required to create a process.
        PROCESS_CREATE_THREAD = 0x0002, //  Required to create a thread.
        PROCESS_DUP_HANDLE = 0x0040, // Required to duplicate a handle using DuplicateHandle.
        PROCESS_QUERY_INFORMATION = 0x0400, //  Required to retrieve certain information about a process, such as its token, exit code, and priority class (see OpenProcessToken, GetExitCodeProcess, GetPriorityClass, and IsProcessInJob).
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000, //  Required to retrieve certain information about a process (see QueryFullProcessImageName). A handle that has the PROCESS_QUERY_INFORMATION access right is automatically granted PROCESS_QUERY_LIMITED_INFORMATION. Windows Server 2003 and Windows XP/2000:  This access right is not supported.
        PROCESS_SET_INFORMATION = 0x0200, //    Required to set certain information about a process, such as its priority class (see SetPriorityClass).
        PROCESS_SET_QUOTA = 0x0100, //  Required to set memory limits using SetProcessWorkingSetSize.
        PROCESS_SUSPEND_RESUME = 0x0800, // Required to suspend or resume a process.
        PROCESS_TERMINATE = 0x0001, //  Required to terminate a process using TerminateProcess.
        PROCESS_VM_OPERATION = 0x0008, //   Required to perform an operation on the address space of a process (see VirtualProtectEx and WriteProcessMemory).
        PROCESS_VM_READ = 0x0010, //    Required to read memory in a process using ReadProcessMemory.
        PROCESS_VM_WRITE = 0x0020, //   Required to write to memory in a process using WriteProcessMemory.
        DELETE = 0x00010000, // Required to delete the object.
        READ_CONTROL = 0x00020000, //   Required to read information in the security descriptor for the object, not including the information in the SACL. To read or write the SACL, you must request the ACCESS_SYSTEM_SECURITY access right. For more information, see SACL Access Right.
        SYNCHRONIZE = 0x00100000, //    The right to use the object for synchronization. This enables a thread to wait until the object is in the signaled state.
        WRITE_DAC = 0x00040000, //  Required to modify the DACL in the security descriptor for the object.
        WRITE_OWNER = 0x00080000, //    Required to change the owner in the security descriptor for the object.
        STANDARD_RIGHTS_REQUIRED = 0x000f0000,
        PROCESS_ALL_ACCESS = STANDARD_RIGHTS_REQUIRED | SYNCHRONIZE | 0xFFF //    All possible access rights for a process object.
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct PROCESS_INFORMATION
    {
        internal IntPtr hProcess;
        internal IntPtr hThread;
        internal uint dwProcessId;
        internal uint dwThreadId;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct STARTUPINFO
    {
        internal uint cb;
        internal IntPtr lpReserved;
        internal IntPtr lpDesktop;
        internal IntPtr lpTitle;
        internal uint dwX;
        internal uint dwY;
        internal uint dwXSize;
        internal uint dwYSize;
        internal uint dwXCountChars;
        internal uint dwYCountChars;
        internal uint dwFillAttribute;
        internal uint dwFlags;
        internal ushort wShowWindow;
        internal ushort cbReserved2;
        internal IntPtr lpReserved2;
        internal IntPtr hStdInput;
        internal IntPtr hStdOutput;
        internal IntPtr hStdError;
    }

    [Flags]
    private enum ProcessCreationFlags : uint
    {
        NONE = 0,
        CREATE_BREAKAWAY_FROM_JOB = 0x01000000,
        CREATE_DEFAULT_ERROR_MODE = 0x04000000,
        CREATE_NEW_CONSOLE = 0x00000010,
        CREATE_NEW_PROCESS_GROUP = 0x00000200,
        CREATE_NO_WINDOW = 0x08000000,
        CREATE_PROTECTED_PROCESS = 0x00040000,
        CREATE_PRESERVE_CODE_AUTHZ_LEVEL = 0x02000000,
        CREATE_SECURE_PROCESS = 0x00400000,
        CREATE_SEPARATE_WOW_VDM = 0x00000800,
        CREATE_SHARED_WOW_VDM = 0x00001000,
        CREATE_SUSPENDED = 0x00000004,
        CREATE_UNICODE_ENVIRONMENT = 0x00000400,
        DEBUG_ONLY_THIS_PROCESS = 0x00000002,
        DEBUG_PROCESS = 0x00000001,
        DETACHED_PROCESS = 0x00000008,
        EXTENDED_STARTUPINFO_PRESENT = 0x00080000,
        INHERIT_PARENT_AFFINITY = 0x00010000
    }
    #endregion
    
    public static uint Start(string exePath, string commandline, string dir, bool hidden, string outputLog, string errorLog, out IntPtr handle, out IntPtr handleStdOut, out IntPtr handleStdErr)
    {
        SECURITY_ATTRIBUTES sa = new SECURITY_ATTRIBUTES();
        sa.nLength = Marshal.SizeOf<SECURITY_ATTRIBUTES>();
        sa.lpSecurityDescriptor = IntPtr.Zero;
        sa.bInheritHandle = 1;
        
        handle = IntPtr.Zero;
        handleStdOut = IntPtr.Zero;
        handleStdErr = IntPtr.Zero;
        if (outputLog != null)
            handleStdOut = CreateFile(outputLog, EFileAccess.FILE_WRITE_DATA, EFileShare.Read | EFileShare.Write, sa, ECreationDisposition.CreateAlways, EFileAttributes.Normal, IntPtr.Zero);
        if (errorLog != null)
            handleStdErr = CreateFile(errorLog, EFileAccess.FILE_WRITE_DATA, EFileShare.Read | EFileShare.Write, sa, ECreationDisposition.CreateAlways, EFileAttributes.Normal, IntPtr.Zero);
        
        ProcessCreationFlags flags = hidden ? ProcessCreationFlags.CREATE_NO_WINDOW : ProcessCreationFlags.NONE;
        STARTUPINFO startupinfo = new STARTUPINFO
        {
            cb = (uint)Marshal.SizeOf<STARTUPINFO>(),
            dwFlags = 0x00000100,
            hStdInput = IntPtr.Zero,
            hStdOutput = handleStdOut,
            hStdError = handleStdErr
        };
        PROCESS_INFORMATION processinfo = new PROCESS_INFORMATION();
        if (!CreateProcessW(exePath, commandline, IntPtr.Zero, IntPtr.Zero, true, flags, IntPtr.Zero, dir, ref startupinfo, ref processinfo))
        {
            throw new Win32Exception();
        }
        handle = OpenProcess(ProcessAccessRights.PROCESS_ALL_ACCESS, false, processinfo.dwProcessId);

        return processinfo.dwProcessId;
    }
    
    public static bool Alive(IntPtr handle) {
        uint ret = WaitForSingleObject(handle, 0);
        return ret == WAIT_TIMEOUT;
    }

    public static int KillProcess(uint pid)
    {
        IntPtr handle = OpenProcess(ProcessAccessRights.PROCESS_ALL_ACCESS, false, pid);

        if (handle == IntPtr.Zero)
        {
            return -1;
        }
        if (!TerminateProcess(handle, 0))
        {
            throw new Win32Exception();
        }
        if (!CloseHandle(handle))
        {
            throw new Win32Exception();
        }

        return 0;
    }
}

}