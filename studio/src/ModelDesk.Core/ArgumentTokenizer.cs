using System.Text;

namespace ModelDesk.Core;

/// <summary>Splits an advanced-arguments field into literal process arguments; never evaluates a shell.</summary>
public static class ArgumentTokenizer
{
    public static IReadOnlyList<string> Parse(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        if (text.Length > 1_048_576) throw new ArgumentException("Arguments exceed 1 MiB.");
        var result = new List<string>();
        var token = new StringBuilder();
        char quote = '\0';
        var started = false;
        for (var index = 0; index < text.Length; index++)
        {
            var c = text[index];
            if (c == '\0' || c is '\r' or '\n') throw new ArgumentException("Use a single line without NUL characters.");
            if (c == '\\' && quote != '\'')
            {
                var count = 1;
                while (index + 1 < text.Length && text[index + 1] == '\\') { count++; index++; }
                if (index + 1 < text.Length && text[index + 1] == '"')
                {
                    token.Append('\\', count / 2);
                    index++;
                    if (count % 2 == 1) token.Append('"');
                    else quote = quote == '"' ? '\0' : '"';
                }
                else token.Append('\\', count);
                started = true;
            }
            else if (c is '\'' or '"' && (quote == '\0' || quote == c))
            {
                quote = quote == c ? '\0' : c;
                started = true;
            }
            else if (char.IsWhiteSpace(c) && quote == '\0')
            {
                if (started) { result.Add(token.ToString()); token.Clear(); started = false; }
            }
            else { token.Append(c); started = true; }
        }
        if (quote != '\0') throw new ArgumentException("A quoted argument is not closed.");
        if (started) result.Add(token.ToString());
        if (result.Count > 256) throw new ArgumentException("Too many arguments (maximum 256).");
        return result;
    }
}
