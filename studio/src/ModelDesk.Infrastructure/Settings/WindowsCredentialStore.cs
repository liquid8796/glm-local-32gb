using System.ComponentModel;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using ModelDesk.Core;

namespace ModelDesk.Infrastructure.Settings;

public sealed class WindowsCredentialStore : ICredentialStore
{
    private readonly string _path;
    private readonly bool _useEnvironment;
    public WindowsCredentialStore(string? stateDirectory = null, bool useEnvironment = true)
    {
        _path = Path.Combine(LocalFiles.StateDirectory(stateDirectory), "huggingface.token.dpapi");
        _useEnvironment = useEnvironment;
    }

    public async Task<string?> GetTokenAsync(CancellationToken cancellationToken = default)
    {
        var environment = _useEnvironment ? Environment.GetEnvironmentVariable("HF_TOKEN") : null;
        if (!string.IsNullOrWhiteSpace(environment)) return Validate(environment);
        if (!File.Exists(_path)) return null;
        var encrypted = await LocalFiles.ReadBoundedAsync(_path, 65536, cancellationToken);
        var plain = Transform(encrypted, protect: false);
        try { return Validate(Encoding.UTF8.GetString(plain)); }
        finally { CryptographicOperations.ZeroMemory(plain); }
    }

    public async Task SetTokenAsync(string? token, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        if (string.IsNullOrWhiteSpace(token))
        {
            LocalFiles.CheckPath(_path);
            if (File.Exists(_path)) File.Delete(_path);
            return;
        }
        var plain = Encoding.UTF8.GetBytes(Validate(token));
        try { await LocalFiles.AtomicWriteAsync(_path, Transform(plain, protect: true), cancellationToken); }
        finally { CryptographicOperations.ZeroMemory(plain); }
    }

    private static string Validate(string token)
    {
        token = token.Trim();
        if (token.Length is < 1 or > 8192 || token.Any(char.IsControl))
            throw new ArgumentException("The authentication token is empty, too long, or contains control characters.");
        return token;
    }

    [StructLayout(LayoutKind.Sequential)] private struct Blob { public int Length; public IntPtr Data; }
    [DllImport("crypt32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool CryptProtectData(ref Blob input, string description, IntPtr entropy, IntPtr reserved,
        IntPtr prompt, uint flags, out Blob output);
    [DllImport("crypt32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool CryptUnprotectData(ref Blob input, IntPtr description, IntPtr entropy, IntPtr reserved,
        IntPtr prompt, uint flags, out Blob output);
    [DllImport("kernel32.dll")] private static extern IntPtr LocalFree(IntPtr value);

    private static byte[] Transform(byte[] bytes, bool protect)
    {
        if (!OperatingSystem.IsWindows()) throw new PlatformNotSupportedException("Credential encryption requires Windows current-user DPAPI.");
        var input = new Blob { Length = bytes.Length, Data = Marshal.AllocHGlobal(bytes.Length) };
        var output = new Blob();
        try
        {
            Marshal.Copy(bytes, 0, input.Data, bytes.Length);
            var success = protect
                ? CryptProtectData(ref input, "ModelDesk Hugging Face token", IntPtr.Zero, IntPtr.Zero, IntPtr.Zero, 1, out output)
                : CryptUnprotectData(ref input, IntPtr.Zero, IntPtr.Zero, IntPtr.Zero, IntPtr.Zero, 1, out output);
            if (!success) throw new Win32Exception(Marshal.GetLastWin32Error(), "Windows could not protect or open the current-user credential.");
            var result = new byte[output.Length];
            Marshal.Copy(output.Data, result, 0, result.Length);
            return result;
        }
        finally
        {
            Marshal.Copy(new byte[bytes.Length], 0, input.Data, bytes.Length);
            Marshal.FreeHGlobal(input.Data);
            if (output.Data != IntPtr.Zero)
            {
                if (!protect && output.Length > 0) Marshal.Copy(new byte[output.Length], 0, output.Data, output.Length);
                LocalFree(output.Data);
            }
        }
    }
}
