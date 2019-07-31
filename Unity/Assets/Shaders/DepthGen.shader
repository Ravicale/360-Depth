Shader "Custom/UnlitParallaxOcclusionMapping" {
	Properties {
		_Image ("Image", 2D) = "white" {}
		_HeightMap ("Depth", 2D) = "bump" {}  
		_Depth ("Depth", Range (0.0001, 0.25)) = 0.08
		_InterpSamples ("Interpolation Samples", Range(1, 5)) = 3
		_Samples ("Samples", Range (1, 255)) = 24
	}

	SubShader {
		Pass {
			Tags { "LightMode" = "ForwardBase" "RenderType" = "Opaque"}
			CGPROGRAM
			#pragma vertex vertex_shader
			#pragma fragment pixel_shader
			#pragma target 3.0

			//Leave one and only one of these uncommented.
			//#define STEEP_PARALLAX_MAPPING
			//#define PARALLAX_OCCLUSION_MAPPING
			#define RELIEF_MAPPING

			sampler2D _Image; 
			sampler2D _HeightMap;
			float3 _CameraOffset = float3(0.0, 0.0, 0.0);
			float _Depth;
			float _InterpSamples;
			float _Samples;

			struct vertIn {
				float4 position : POSITION;
				float4 normal   : NORMAL;
				float4 tangent  : TANGENT;
				float2 texCoord : TEXCOORD0;
			};
			
			struct vertOut {
				float4 pos      : SV_POSITION;
				float2 texCoord : TEXCOORD0;
				float4 posWorld : TEXCOORD1;
				float3 viewDir  : TEXCOORD2;
			};

			vertOut vertex_shader( vertIn vIn ) {
				vertOut vOut; 
				//Get information in 'tangent space' (relative to the surface).
				vOut.posWorld = mul(unity_ObjectToWorld, vIn.position);
				fixed3 worldNormal = normalize(mul(vIn.normal.xyz, unity_WorldToObject));
				fixed3 worldTangent =  normalize(mul(unity_ObjectToWorld, vIn.tangent.xyz ));
				fixed3 worldBitangent = cross(worldNormal, worldTangent) * vIn.tangent.w; 

				vOut.pos = UnityObjectToClipPos(vIn.position);
				vOut.texCoord = vIn.texCoord;

				float3x3 worldToTan = float3x3(
					worldTangent,
					worldBitangent,
					worldNormal
				);

				//Tangent space view direction calculated and interpolated between vertices for performance.
				//Quality loss is minor for the current use case.
				float3 worldViewDir = normalize((_WorldSpaceCameraPos.xyz + _CameraOffset) - vOut.posWorld.xyz);
				vOut.viewDir = mul(worldToTan, worldViewDir);

				return vOut;
			}

			float4 pixel_shader( vertOut fIn ): SV_TARGET {
				//'Vertical' distance covered by each step.
				float stepSize = 1.0 / _Samples;

				//Current 'vertical' depth of sample.
				float currHeight = 1.0;

				//'Horizontal' distance in tex coords covered by each step.
				float2 deltaOffset = (fIn.viewDir.xy / fIn.viewDir.z) * (_Depth / _Samples);

				//'Horizontal' position of heightmap to sample.
				float2 currTexCoord = fIn.texCoord;
				float2 prevTexCoord = 0.0;

				//Current and previous heights of texture.
				float currSurfaceHeight = tex2D( _HeightMap, currTexCoord);

				float fParallaxAmount = 0.0;

				//Cast ray through heightmap and see what we hit.
				//step < 255 done to tell the compiler how to unroll the loop, leave it there.
				for (int step = 0; step < _Samples && step < 255 && currSurfaceHeight < currHeight; step++) {
					//Check for intersection.
					currTexCoord -= deltaOffset; //Travel across heightmap and resample.
					currSurfaceHeight = tex2D( _HeightMap, currTexCoord).r;
					currHeight -= stepSize;
				}

				#if defined(STEEP_PARALLAX_MAPPING)
					//Returns whatever garbage we get.
					prevTexCoord = currTexCoord + deltaOffset;
					return tex2D(_Image, prevTexCoord);
				#endif

				#if defined(PARALLAX_OCCLUSION_MAPPING)
					//Interpolates between current step (too far) and previous step (not far enough)
					prevTexCoord = currTexCoord + deltaOffset;
					float lastHeight = currSurfaceHeight - currHeight;
					float prevHeight = tex2D(_HeightMap, prevTexCoord).r - currHeight + stepSize;
					float weight = lastHeight / (lastHeight + prevHeight);
					return tex2D(_Image, lerp(prevTexCoord, currTexCoord, weight));
				#endif	

				#if defined(RELIEF_MAPPING)
					//Uses binary search to refine sampled texture.
					for (int step = 0; step < 6 && step < _InterpSamples; step++) {
						//Half distance travelled with each step.
						deltaOffset *= 0.5;
						stepSize *= 0.5;

						if (currHeight > currSurfaceHeight) {
							currTexCoord += deltaOffset;
							currSurfaceHeight = tex2D(_HeightMap, currTexCoord).r;
							currHeight += stepSize;
						} else if (currHeight < currSurfaceHeight) {
							currTexCoord -= deltaOffset;
							currSurfaceHeight = tex2D(_HeightMap, currTexCoord).r;
							currHeight -= stepSize;
						} else { //On the off chance we somehow achieve equality.
							break;
						}
					}

					return tex2D(_Image, currTexCoord);
				#endif
			}           
			ENDCG
		}
	}
}