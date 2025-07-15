using System;
using System.Net.Http;
using System.Net.Http.Json;
using System.Threading.Tasks;
using System.Collections.Generic;

public class InputData
{
    public int Age { get; set; }
    public float SleepDuration { get; set; }
    public int StressLevel { get; set; }
    public int PhysicalActivityLevel { get; set; }
}

class Program
{
    static async Task Main(string[] args)
    {
        Console.WriteLine("=== Sleep Health Predictor ===");

        Console.Write("Enter Age: ");
        int age = int.Parse(Console.ReadLine());

        Console.Write("Enter Sleep Duration (in hours): ");
        float sleep = float.Parse(Console.ReadLine());

        Console.Write("Enter Stress Level (1-10): ");
        int stress = int.Parse(Console.ReadLine());

        Console.Write("Enter Physical Activity Level (1-10): ");
        int activity = int.Parse(Console.ReadLine());

        var input = new InputData
        {
            Age = age,
            SleepDuration = sleep,
            StressLevel = stress,
            PhysicalActivityLevel = activity
        };

        using var client = new HttpClient();
        var response = await client.PostAsJsonAsync("http://127.0.0.1:8000/predict", input);

        if (response.IsSuccessStatusCode)
        {
            var result = await response.Content.ReadFromJsonAsync<Dictionary<string, string>>();
            Console.WriteLine($"\n✅ Prediction: {result["prediction"]}");
        }
        else
        {
            Console.WriteLine($"❌ Error: {response.StatusCode}");
        }
    }
    
}