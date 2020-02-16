using System;
using System.ComponentModel;
using System.Diagnostics;
using System.Runtime.InteropServices;

namespace OpenSee {

// Based on https://github.com/josh4364/IL2cppStartProcess/blob/master/StartExternalProcess.cs

public class OpenSeeProcessInterface {
    #region DllImport
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
    
    public static uint Start(string exePath, string commandline, string dir, bool hidden, out IntPtr handle)
    {
        handle = IntPtr.Zero;
        ProcessCreationFlags flags = hidden ? ProcessCreationFlags.CREATE_NO_WINDOW : ProcessCreationFlags.NONE;
        STARTUPINFO startupinfo = new STARTUPINFO
        {
            cb = (uint)Marshal.SizeOf<STARTUPINFO>()
        };
        PROCESS_INFORMATION processinfo = new PROCESS_INFORMATION();
        if (!CreateProcessW(exePath, commandline, IntPtr.Zero, IntPtr.Zero, false, flags, IntPtr.Zero, dir, ref startupinfo, ref processinfo))
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