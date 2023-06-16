using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using API.Data;
using chatbot_server.Models;
using NuGet.Protocol.Plugins;
using Autobot.Models;
using System.Security.Cryptography;
using System.Text;
using NuGet.Common;
using Microsoft.AspNetCore.Cors;

namespace API.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    [EnableCors("All")]
    public class UsersController : ControllerBase
    {
        private readonly APIContext _context;
        private SHA256 _sha256;

        public UsersController(APIContext context)
        {
            _context = context;
            _sha256 = SHA256.Create();
        }

        // POST: api/Users/login
        [HttpPost("login")]
        public async Task<ActionResult<Users>> Login([FromBody] LoginRequest loginRequest)
        {
            var users = await _context.Users.FirstOrDefaultAsync(u => u.Username == loginRequest.Username && u.HashPassword == Convert.ToBase64String(_sha256.ComputeHash(Encoding.UTF8.GetBytes(loginRequest.Password))));

            if (users == null)
            {
                return NotFound();
            }

            return users;
        }

        // POST: api/Users/register
        [HttpPost("register")]
        public async Task<ActionResult<Users>> Register([FromBody] RegisterRequest registerRequest)
        {
            // Check if the username already exists
            var existingUser = await _context.Users.FirstOrDefaultAsync(u => u.Username == registerRequest.Username);
            if (existingUser != null)
            {
                return Conflict("Username already exists");
            }
            string hashedPassword = Convert.ToBase64String(_sha256.ComputeHash(Encoding.UTF8.GetBytes(registerRequest.Password)));
            // Create a new user object
            var newUser = new Users
            {
                Username = registerRequest.Username,
                HashPassword = hashedPassword
            };
            // Add the new user to the database
            _context.Users.Add(newUser);
            await _context.SaveChangesAsync();
            return Ok(newUser);
        }
    }
}
