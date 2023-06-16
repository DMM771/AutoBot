using Microsoft.AspNetCore.Mvc;
using System.Diagnostics;
using Microsoft.AspNetCore.Cors;



namespace Chat_Api.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    [EnableCors("All")]
    public class ChatController : ControllerBase
    {

        [HttpGet("message")]
        public IActionResult Get(string message)
        {
            string pythonScriptPath = Path.GetFullPath("chat.py");
            string scriptDirectory = Path.GetDirectoryName(pythonScriptPath);
            ProcessStartInfo startInfo = new ProcessStartInfo
            {
                FileName = "python3",
                Arguments = $"\"chat.py\" \"{message}\"",
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true,
                WorkingDirectory = scriptDirectory
            };

            using (Process process = Process.Start(startInfo))
            {
                using (StreamReader reader = process.StandardOutput)
                {
                    string output = reader.ReadToEnd();
                    process.WaitForExit();
                    using (StreamReader errorReader = process.StandardError)
                    {
                        string errors = errorReader.ReadToEnd();
                        Console.WriteLine("Error output: " + errors);
                    }
                    var l = new List<TextResponse>();
                    l.Add(new TextResponse() { response = output });
                    return Ok(l);
                }
            }
        }

        [HttpPost("image")]
        public async Task<IActionResult> GetImage(IFormFile image)
        {
            // Save the image to a file
            string imagePath = "temp.jpg";
            using (var stream = new FileStream(imagePath, FileMode.Create))
            {
                await image.CopyToAsync(stream);
            }
            string pythonScriptPath = Path.GetFullPath("Number_Detector_Final.py");
            string scriptDirectory = Path.GetDirectoryName(pythonScriptPath);
            ProcessStartInfo startInfo = new ProcessStartInfo
            {
                FileName = "python3",
                Arguments = $"\"Number_Detector_Final_Cleaned.py\" \"{imagePath}\"",
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true,
                WorkingDirectory = scriptDirectory
            };
            using (Process process = Process.Start(startInfo))
            {
                using (StreamReader reader = process.StandardOutput)
                {
                    string output = reader.ReadToEnd();
                    process.WaitForExit();
                    using (StreamReader errorReader = process.StandardError)
                    {
                       string errors = errorReader.ReadToEnd();
                        Console.WriteLine("Error output: " + errors);
                    }
                    // Delete the temporary image file
                    if (System.IO.File.Exists(imagePath))
                    {
                        System.IO.File.Delete(imagePath);
                    }
                    string[] substrings = output.Split('#');
                    int missing_numbers = substrings[1].Count(c => c == 'x');
                    if (missing_numbers==1)
                    {
                        output = substrings[1];
                        var l = new List<TextResponse>();
                        String[] results = new String[10];
                        for (int i = 0; i <= 9; i++)
                        {
                            string newText = output.Replace('x', i.ToString()[0]);
                            var data = new Dictionary<string, string>
                            {
                                { "resource_id", "053cea08-09bc-40ec-8f7a-156f0677aff3" }, // the resource id
                                { "q", newText },
                            };
                            var queryString = new FormUrlEncodedContent(data);
                            var client = new HttpClient();
                            var url = "https://data.gov.il/api/3/action/datastore_search";
                            var response = await client.GetAsync($"{url}?{await queryString.ReadAsStringAsync()}");
                            response.EnsureSuccessStatusCode();
                            results[i] = await response.Content.ReadAsStringAsync();
                            int count = 0;
                            int index = results[i].IndexOf(newText);

                            while (index != -1)
                            {
                                count++;
                                index = results[i].IndexOf(newText, index + newText.Length);
                            }
                            if (count != 4)
                            {
                                 results[i] = null;
                            }
                        }
                        String detected_color = substrings[0];
                        List<string> colors = new List<string>();
                        //get color
                        switch (detected_color)
                        {
                            case "grey":
                                colors.Add("לבן");
                                colors.Add("כסוף");
                                colors.Add("כסף");
                                colors.Add("שחור");
                                colors.Add("אפור");
                                break;
                            case "black":
                                colors.Add("שחור");
                                colors.Add("אפור");
                                break;
                            case "brown":
                                colors.Add("חום");
                                colors.Add("בז");
                                break;
                            case "blue":
                                colors.Add("כחול");
                                colors.Add("תכלת");
                                break;
                            case "white":
                            case "silver":
                                colors.Add("לבן");
                                colors.Add("כסף");
                                colors.Add("כסוף");
                                colors.Add("אפור");
                                colors.Add("בז");
                                break;
                            case "yello":
                            case "orange":
                                colors.Add("צהוב");
                                colors.Add("כתום");
                                colors.Add("זהב");
                                colors.Add("בז");
                                break;
                            case "red":
                                colors.Add("אדום");
                                break;
                            case "green":
                                colors.Add("ירוק");
                                break;
                        }
                        foreach (string color in colors)
                        {
                           for (int i = 0; i <= 9; i++)
                            {
                                if (results[i]!=null && results[i].IndexOf(color) != -1)
                                    l.Add(new TextResponse() { response = results[i] });
                            }
                        }
                        return Ok(l);
                    }
                    else if (missing_numbers==0)
                    {
                        var data = new Dictionary<string, string>
                        {
                            { "resource_id", "053cea08-09bc-40ec-8f7a-156f0677aff3" }, // the resource id
                            { "q", substrings[1] },
                        };
                        var queryString = new FormUrlEncodedContent(data);
                        var client = new HttpClient();
                        var url = "https://data.gov.il/api/3/action/datastore_search";
                        var response = await client.GetAsync($"{url}?{await queryString.ReadAsStringAsync()}");
                        response.EnsureSuccessStatusCode();
                        var result = await response.Content.ReadAsStringAsync();
                        var l = new List<TextResponse>();
                        l.Add(new TextResponse() { response = result });
                        return Ok(l);
                    }
                    else
                    {
                        var l = new List<TextResponse>();
                        return Ok(l);
                    }

                }
            }
        }

        [HttpGet("number")]
        public async Task<IActionResult> GetDetails(string number)
        {
            var data = new Dictionary<string, string>
                        {
                            { "resource_id", "053cea08-09bc-40ec-8f7a-156f0677aff3" }, // the resource id
                            { "q", number },
                        };
            var queryString = new FormUrlEncodedContent(data);
            var client = new HttpClient();
            var url = "https://data.gov.il/api/3/action/datastore_search";
            var response = await client.GetAsync($"{url}?{await queryString.ReadAsStringAsync()}");
            response.EnsureSuccessStatusCode();
            var result = await response.Content.ReadAsStringAsync();
            var l = new List<TextResponse>();
            l.Add(new TextResponse() { response = result });
            return Ok(l);
        }
    }
}